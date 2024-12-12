import pandas as pd
import numpy as np
import pickle
from numpy import zeros
import tensorflow as tf
import random as python_random
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from utils.procesamiento import dir_data_embedding, dir_data_proc
from gensim.models import fasttext


def initialize_keras():
    """ Configurates keras session to make results replicable"""
    os.environ['PYTHONHASHSEED']=str(2023)

    np.random.seed(2023)
    python_random.seed(2023)
    tf.random.set_seed(2023)

    keras.utils.set_random_seed(2023)
    tf.config.experimental.enable_op_determinism()

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session()
    K.set_session(sess)

def split_data(data, model_name, partition):
    
    # Definimos las columnas para las etiquetas
    cols = {
        'CAENES_2d': 'caenes_2d',
        'CAENES_4d': 'caenes_4d',
        'CIUO_2d': 'ciuo_2d',
        'CIUO_4d': 'ciuo_4d'
    }

    # Definir las características (inputs) y la etiqueta
    observations = data[['glosa_ocupacion', 'glosa_tareas', 'activ_principal']]
    labels = data[cols[model_name]]

    X_train, X_test, y_train, y_test = train_test_split(observations, labels, test_size=partition, random_state=2023, stratify=labels)

    return X_train, X_test, y_train, y_test

def load_embeddings():
    """ load pretrained embeddings """

    print(f"Loading model embeddings...")
    embeddings = fasttext.load_facebook_model(dir_data_embedding  / 'embeddings-l-model.bin')
    dim_embeddings = 300
    
    #embeddings = fasttext.load_facebook_model(dir_data_embedding  / 'embeddings-s-model.bin')
    #dim_embeddings = 30
    return embeddings, dim_embeddings


def tokenize_data(X_train, X_test, text, padding_len=50):

    X_train = X_train[text]
    X_test = X_test[text]

    # Inicializar un tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)                  # ajustando vocab

    # Convertir las secuencias de texto en secuencias de tokens
    X_train = tokenizer.texts_to_sequences(X_train)  # obteniendo vocab train
    X_test = tokenizer.texts_to_sequences(X_test)    # obteniendo vocab test

    # Realizar padding en las secuencias  
    X_train_pad = pad_sequences(X_train, maxlen=padding_len)
    X_test_pad = pad_sequences(X_test, maxlen=padding_len)

    return X_train_pad, X_test_pad, tokenizer

def create_matrix_embeddings(tokenizer, embeddings, n_vocab, dim_model=300):
    """
    Funcion que crea matriz de embeddings con nuestro vocabulario disponible en las glosas

    INPUT: 
    - tokenizer: tokenizer que contiene vocabulario de train set
    - embeddings: modelo de embeddings fasttext de Jorge Perez
    - n_vocab: largo del vocaulario a usar (de tokenizer)
    - dim_model: dimension modelo de embeddings

    OUTPUT:
    - embedding_matrix: matriz con embeddings 
    """
    # matrix with zeros
    matrix_embeddings = zeros((n_vocab, int(dim_model)))

    n = 0
    for word, i in tokenizer.word_index.items():

        # Si la palabra esta en el modelo de embeddings
        if word in embeddings.wv: 
            # Guardar embedding
            matrix_embeddings[i] = embeddings.wv[word] 
            n += 1 

    print(n)
    return(matrix_embeddings)

def save_pkl(file, file_name):
    """funcion para guardar archivos pkl"""

    with open(file_name +'.pkl', 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(X_train_var):
    """funcion para cargar archivos pkl"""

    with open(X_train_var +'.pkl', 'rb') as f:
        X_train_var = pickle.load(f)

    return X_train_var

def read_data_models(model_name):
    """
    Carga y filtra los datos de acuerdo a la desagregación especificada.

    Args:
        model_name (str): El tipo de datos que se desea cargar o filtrar.

    Returns:
        pd.DataFrame: El DataFrame filtrado según la condición especificada.
    """
    # Cargar los datos consolidados
    if model_name == 'CAENES_2d' or model_name == 'CAENES_4d':
        df = pd.read_parquet(dir_data_proc / 'data_procesada_caenes.parquet')
    else:
        df = pd.read_parquet(dir_data_proc / 'data_procesada_ciuo.parquet')

    filtros = {
        'CAENES_2d': 'caenes_2d',
        'CAENES_4d': 'caenes_4d',
        'CIUO_2d': 'ciuo_2d',
        'CIUO_4d': 'ciuo_4d'
    }

    # Aplicar el filtro correspondiente
    if model_name in filtros:
        # Dejar clases que tengan por lo menos dos ejemplos
        df = df.groupby(filtros[model_name]).filter(lambda x: len(x) >= 2)
        df = df.reset_index(drop=True)
        return df
    else:
        raise ValueError(f"Tipo de datos no reconocido: {model_name}")