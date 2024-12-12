import pandas as pd
import numpy as np
import sys
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import string
from utils.procesamiento import dir_input
from keras.models import load_model
from utils.model import load_pkl
from transformers import pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences

def process_glosa(text):
    
    # Eliminar signos de puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Pasar a minúsculas y quitar espacios extras
    text = text.lower().strip()
    
    # Eliminar stop words
    text = text.split()
    stop_words = set(stopwords.words("spanish"))
    text = [word for word in text if word not in stop_words]
    text = " ".join(text)
    
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # Eliminar acentos
    patterns = [r'[áàäâåã]', r'[éèëê]', r'[íïîì]', r'[óôöòõô]', r'[úûùü]']
    replacements = ['a', 'e', 'i', 'o', 'u']
    
    for i, pattern in enumerate(patterns):
        text = re.sub(pattern, replacements[i], text)
    
    return text

def load_data():
    
    input_path = dir_input / 'muestra_5_casen_2022.dta'
    
    data = pd.read_stata(input_path, convert_categoricals=False)
    
    data = data[['folio', 'id_persona', 'oficio4_08', 'rama4', 'o9a', 'o9b', 'o24']]
    
    # Renombrar columnas
    data = data.rename(columns={'o9a': 'glosa_ocupacion',
                                'o9b': 'glosa_tareas',
                                'o24': 'activ_principal'})
    
    # Filtrar filas donde glosa_ocupacion, glosa_tareas o activ_principal están vacíos o NaN
    data = data[(data['glosa_ocupacion'] != "") & 
                (data['glosa_tareas'] != "") & 
                (data['activ_principal'] != "")]
    
    # Limpiar glosas
    data['glosa_ocupacion'] = data['glosa_ocupacion'].map(lambda x: process_glosa(x) if pd.notna(x) else "")
    data['glosa_tareas'] = data['glosa_tareas'].map(lambda x: process_glosa(x) if pd.notna(x) else "")
    data['activ_principal'] = data['activ_principal'].map(lambda x: process_glosa(x) if pd.notna(x) else "")
    
    data = data.reset_index(drop = True)
    
    return data

def predict_with_model(model_name, data):
    
    print(f"\nPredicting data: {model_name}")
    
    if model_name == 'CAENES_4d':

        # Cargar el modelo entrenado
        model = load_model(f"../models/modelo_2_{model_name}_lstm.h5")
        
        # Cargar los artefactos
        tokenizer_tareas = load_pkl(f'../models/artifacts/modelo_2_tokenizer_tareas_{model_name}')
        tokenizer_ocupacion = load_pkl(f'../models/artifacts/modelo_2_tokenizer_ocupacion_{model_name}')
        tokenizer_activ_principal = load_pkl(f'../models/artifacts/modelo_2_tokenizer_activ_principal_{model_name}')
        label_encoder = load_pkl(f'../models/artifacts/modelo_2_label_encoder_{model_name}')
            
        # Tokenizar
        n_pad = max([len(str(x).split(" ")) for col in ['glosa_tareas', 'glosa_ocupacion', 'activ_principal'] 
                for x in data[col].dropna().values])
        
        data_token_tareas= tokenizer_tareas.texts_to_sequences(data['glosa_tareas'])
        data_pad_tareas = pad_sequences(data_token_tareas, maxlen= n_pad)
        
        data_token_ocupacion= tokenizer_ocupacion.texts_to_sequences(data['glosa_ocupacion'])
        data_pad_ocupacion = pad_sequences(data_token_ocupacion, maxlen= n_pad)
        
        data_token_activ= tokenizer_activ_principal.texts_to_sequences(data['activ_principal'])
        data_pad_activ = pad_sequences(data_token_activ, maxlen= n_pad)
        
        # Realizar la predicción
        predictions = model.predict({
            "input_glosa1": data_pad_tareas,
            "input_glosa2": data_pad_ocupacion,
            "input_glosa3": data_pad_activ
        })
        
        # Obtener las clases predichas
        y_pred_probas_max = np.argmax(predictions, axis=1)
        y_pred_class = label_encoder.inverse_transform(y_pred_probas_max)
        y_pred_probabilities = np.max(predictions, axis=1)
        
    else:
        
        data['glosa'] = data['glosa_ocupacion'] + ' [SEP] ' + data['glosa_tareas'] + ' [SEP] ' + data['activ_principal']
        
        # Cargar el LabelEncoder
        label_encoder = load_pkl(f'../models/artifacts/modelo_2_label_encoder_{model_name}')
        
        # Crear el pipeline
        text_pipeline = pipeline(
            "text-classification",
            model="../models/sabe_fine_tunning")

        # Generar predicciones
        textos = data['glosa'].tolist()  # Lista de textos
        predicciones = text_pipeline(textos, truncation=True, padding=True, return_all_scores=True)
        
        resultados = []  # Lista para almacenar los resultados

        for i, prediccion in enumerate(predicciones):
            #print(i, prediccion)
            score = [d["score"] for d in prediccion]
            label = [d["label"] for d in prediccion]
            #print(score)
            #print(label)
            score_max_arg = np.argmax(score)
            #print(score_max_arg) 
            score_max = score[score_max_arg]
            label_max = label[score_max_arg]
            label_max_num = int(label_max.split('_')[-1])
            label_max_2 = label_encoder.classes_[label_max_num]
            
            resultados.append({"label": label_max_2, "score": score_max})
            
        resultados_df = pd.DataFrame.from_dict(resultados)
        
        y_pred_class = resultados_df['label']
        y_pred_probabilities = resultados_df['score']
        
    
    # Crear un nuevo DataFrame para los resultados
    results = data

    # Crear una máscara para las filas con valores NaN o vacíos
    mask = results[['glosa_tareas', 'glosa_ocupacion', 'activ_principal']].isna().any(axis=1)

    # Asignar valores NaN a las columnas de predicción para las filas que cumplan la condición
    results.loc[mask, [f'predicted_class_{model_name}', f'predicted_probability_{model_name}']] = np.nan

    # Asignar las predicciones a las filas que no cumplan la condición
    results.loc[~mask, f'predicted_class_{model_name}'] = y_pred_class
    results.loc[~mask, f'predicted_probability_{model_name}'] = y_pred_probabilities
    
    # Calcular los ventiles para las probabilidades predichas
    results[f'predicted_probability_ventile_{model_name}'] = pd.qcut(results[f'predicted_probability_{model_name}'], q=20, labels=False, duplicates='drop') + 1
    
    # Revisar si las clases coinciden hasta cierto ventil
    if model_name == 'CAENES_4d':
        
        filtered_rows = results[f'predicted_probability_ventile_{model_name}'] >= 9
        
        results.loc[filtered_rows, 'match'] = (
        results.loc[filtered_rows, f'predicted_class_{model_name}'].astype(int) == results.loc[filtered_rows, 'rama4']).astype(int)
        
    else:
        
        filtered_rows = results[f'predicted_probability_ventile_{model_name}'] >= 10
        
        results.loc[filtered_rows, 'match'] = (
        results.loc[filtered_rows, f'predicted_class_{model_name}'].astype(int) == results.loc[filtered_rows, 'oficio4_08']).astype(int)
        
    return results

def save_predictions(data, model_name):

    print(f"Saving predictions to output files...")
    
    output_file = f'../data/output/bases_prediccion_casen/predicciones_casen_{model_name}.xlsx'
    data.to_excel(output_file, index=False)

    print(f"Predicciones guardadas en {output_file}")
    
if __name__ == "__main__":

    # retrieve arguments passed from the calling script
    model_name = sys.argv[1]
    
    # load training data
    data = load_data()

    # get model predictions
    all_predictions = predict_with_model(model_name, data)

    # generate output files
    save_predictions(all_predictions, model_name)