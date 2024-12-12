import pandas as pd
import sys
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel, AutoModelForSequenceClassification
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.model import initialize_keras, split_data, tokenize_data, create_matrix_embeddings, save_pkl, load_embeddings, read_data_models
from sklearn.metrics import f1_score, accuracy_score
from utils.procesamiento import dir_data_pred


def make_sabe_predictions(model_name):
    
    data = read_data_models(model_name)
    
    X_train, X_test, y_train, y_test = split_data(data, 'CIUO_4d', 0.3 )
    
    X_test['glosa'] = X_test['glosa_ocupacion'] + ' [SEP] ' + X_test['glosa_tareas'] + ' [SEP] ' + X_test['activ_principal']
    
    # Inicializar el LabelEncoder
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(y_train)
    test_labels = label_encoder.transform(y_test)
    
    # Carga el modelo y el tokenizer desde el directorio guardado
    model = AutoModelForSequenceClassification.from_pretrained("../models/sabe_fine_tunning")
    tokenizer = AutoTokenizer.from_pretrained("../models/sabe_fine_tunning")
    
    # Tokenizar las entradas
    inputs = tokenizer(X_test['glosa'].to_list(), padding=True, truncation=True, return_tensors="pt")

    # Crear un DataLoader para X_test
    test_dataset = TensorDataset(
        inputs["input_ids"],
        inputs["attention_mask"]
    )
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Modo evaluación

    model.eval()

    all_predictions = []
    all_probabilities = []
 
    # Realizar predicciones por lotes
    with torch.no_grad():

        for batch in test_loader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            pred_probabilities = torch.softmax(logits, dim=-1)  # Obtener las probabilidades con softmax
            
            # Convertir las predicciones y probabilidades a numpy arrays o listas
            all_predictions.extend(predictions.numpy())  # Convertir predicciones a numpy
            all_probabilities.extend(pred_probabilities.tolist())  # Usar tolist() para las probabilidades
    
    # Convertir a numpy array para análisis
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Convertir las predicciones a clases
    y_pred_class = label_encoder.inverse_transform(all_predictions)
    y_pred_probabilities = np.max(all_probabilities, axis=1)
    
    # Agregar las predicciones a la base original filtrada
    results = data.loc[y_test.index].copy()
    results['predicted_class'] = y_pred_class
    results['predicted_probability'] = y_pred_probabilities
    
     # Guardar resultados
    results.to_parquet(dir_data_pred / f'predicciones_{model_name}_sabe.parquet')
    
    return results
    
def evaluate_model(df, model_name):
    """
    Calcula y devuelve las métricas accuracy y F1 para un modelo dado.
    
    Args:
        model_name (str): El nombre del modelo.
    
    Returns:
        dict: Diccionario con las métricas de accuracy y F1.
    """
        
    cols = {
        'CIUO_4d': 'ciuo_4d'
    }
    
    # Calcula las métricas
    accuracy = accuracy_score(df[cols[model_name]], df['predicted_class'])
    f1 = f1_score(df[cols[model_name]], df['predicted_class'], average='weighted')
    f1_macro = f1_score(df[cols[model_name]], df['predicted_class'], average='macro')# Puedes cambiar 'weighted' por 'macro' o 'micro' según el problema
    #balanced_accuracy = balanced_accuracy_score(df[cols[model_name]], df['predicted_class'])

    # Imprimir las métricas
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Macro: {f1_macro:.2f}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro    }

if __name__ == "__main__":
    
    model_name = sys.argv[1]
    
    # get model predictions
    predictions = make_sabe_predictions(model_name)

    # evaluate model
    evaluate_model(predictions, model_name)
