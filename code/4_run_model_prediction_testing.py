import sys
import numpy as np
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.model import read_data_models
from utils.procesamiento import dir_data_pred
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def load_test_data(model_name, version):
    with open(f'../models/artifacts/{version}_X_test_tareas_pad_{model_name}.pkl', 'rb') as file:
        X_test_tareas_pad = pickle.load(file)
    with open(f'../models/artifacts/{version}_X_test_ocupacion_pad_{model_name}.pkl', 'rb') as file:
        X_test_ocupacion_pad = pickle.load(file)
    with open(f'../models/artifacts/{version}_X_test_activ_principal_pad_{model_name}.pkl', 'rb') as file:
        X_test_activ_principal_pad = pickle.load(file)
    with open(f'../models/artifacts/{version}_label_encoder_{model_name}.pkl', 'rb') as file:
        label_encoder = pickle.load(file)  
    with open(f'../models/artifacts/{version}_y_test_index_{model_name}.pkl', 'rb') as file:
        y_test_index = pickle.load(file)       
    return X_test_tareas_pad, X_test_ocupacion_pad, X_test_activ_principal_pad, label_encoder, y_test_index

def make_predictions(model_name, version):

    # Cargar el modelo entrenado
    if version == 'modelo_1':
        print(version)
        model = load_model(f"../models/modelo_1_{model_name}_lstm_70_epochs_100.h5")
        print(model.summary())
    else:
        print(version)
        model = load_model(f"../models/modelo_2_{model_name}_lstm.h5")
        print(model.summary())
    
    # Cargar los artefactos
    X_test_tareas_pad, X_test_ocupacion_pad, X_test_activ_principal_pad, label_encoder, y_test_index = load_test_data(model_name, version)
    
    # Cargar la base procesada
    data = read_data_models(model_name)

    # Realizar las predicciones
    predictions = model.predict({
        "input_glosa1": X_test_tareas_pad,
        "input_glosa2": X_test_ocupacion_pad,
        "input_glosa3": X_test_activ_principal_pad
    })
    
    # Convertir las predicciones a clases
    y_pred_probas_max = np.argmax(predictions, axis=1)
    y_pred_class = label_encoder.inverse_transform(y_pred_probas_max)
    y_pred_probabilities = np.max(predictions, axis=1)

     # Agregar las predicciones a la base original filtrada
    results = data.loc[y_test_index].copy()
    results['predicted_class'] = y_pred_class
    results['predicted_probability'] = y_pred_probabilities

    # Guardar resultados
    results.to_parquet(dir_data_pred / f'predicciones_{model_name}_{version}.parquet')
    
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
        "CAENES_2d": 'caenes_2d',
        "CAENES_4d": 'caenes_4d',
        'CIUO_2d': 'ciuo_2d',
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
    version = sys.argv[2]
    
    # get model predictions
    predictions = make_predictions(model_name, version)

    # evaluate model
    evaluate_model(predictions, model_name)
