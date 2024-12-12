# -*- coding: utf-8 -*-

import sys
import numpy as np
from gensim.models import fasttext
from tensorflow import keras
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from utils.model import initialize_keras, split_data, tokenize_data, create_matrix_embeddings, save_pkl, load_embeddings, read_data_models
    
def get_model_artifacts(model_name, embeddings, dim_embeddings):
    
    data = read_data_models(model_name)
    
    cols = {
        'CAENES_2d': 'caenes_2d',
        'CAENES_4d': 'caenes_4d',
        'CIUO_2d': 'ciuo_2d',
        'CIUO_4d': 'ciuo_4d'
    }
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(data, model_name, 0.3)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test) 
    
    # save model artifacts
    save_pkl(label_encoder, f'../models/artifacts/modelo_1_label_encoder_{model_name}')
    save_pkl(y_test.index, f'../models/artifacts/modelo_1_y_test_index_{model_name}')
    save_pkl(y_train.index, f'../models/artifacts/modelo_1_y_train_index_{model_name}')
    save_pkl(y_test_encoded, f'../models/artifacts/modelo_1_y_test_encoded_{model_name}')
    save_pkl(y_train_encoded, f'../models/artifacts/modelo_1_y_train_encoded_{model_name}')
    
    print(f"y_test shape: {y_test.shape}; y_train shape: {y_train.shape}") 

    # Tokenizar y hacer padding para cada glosa
    n_pad = max([len(x.split(" ")) for col in ['glosa_tareas', 'glosa_ocupacion', 'activ_principal'] 
                 for x in X_train[col].values])
    
    # Tokenización de la primera glosa
    X_train_tareas_pad, X_test_tareas_pad, tokenizer_tareas = tokenize_data(X_train, X_test, 'glosa_tareas', padding_len=n_pad)
    save_pkl(tokenizer_tareas, f'../models/artifacts/modelo_1_tokenizer_tareas_{model_name}')

    # Tokenización de la segunda glosa
    X_train_ocupacion_pad, X_test_ocupacion_pad, tokenizer_ocupacion= tokenize_data(X_train, X_test, 'glosa_ocupacion', padding_len=n_pad)
    save_pkl(tokenizer_ocupacion, f'../models/artifacts/modelo_1_tokenizer_ocupacion_{model_name}')

    # Tokenización de la tercera glosa
    X_train_activ_principal_pad, X_test_activ_principal_pad, tokenizer_activ_principal= tokenize_data(X_train, X_test, 'activ_principal', padding_len=n_pad)
    save_pkl(tokenizer_activ_principal, f'../models/artifacts/modelo_1_tokenizer_activ_principal_{model_name}')

    # Guardar artefactos
    save_pkl(X_test_tareas_pad, f'../models/artifacts/modelo_1_X_test_tareas_pad_{model_name}')
    save_pkl(X_train_tareas_pad, f'../models/artifacts/modelo_1_X_train_tareas_pad_{model_name}')
    save_pkl(X_test_ocupacion_pad, f'../models/artifacts/modelo_1_X_test_ocupacion_pad_{model_name}')
    save_pkl(X_train_ocupacion_pad, f'../models/artifacts/modelo_1_X_train_ocupacion_pad_{model_name}')
    save_pkl(X_test_activ_principal_pad, f'../models/artifacts/modelo_1_X_test_activ_principal_pad_{model_name}')
    save_pkl(X_train_activ_principal_pad, f'../models/artifacts/modelo_1_X_train_activ_principal_pad_{model_name}')
    
    # Obtener matrices de embeddings para cada tokenizer
    matrix_embeddings_tareas = create_matrix_embeddings(tokenizer_tareas, embeddings, len(tokenizer_tareas.index_word) + 1, dim_model=dim_embeddings)
    matrix_embeddings_ocupacion = create_matrix_embeddings(tokenizer_ocupacion, embeddings, len(tokenizer_ocupacion.index_word) + 1, dim_model=dim_embeddings)
    matrix_embeddings_activ_principal = create_matrix_embeddings(tokenizer_activ_principal, embeddings, len(tokenizer_activ_principal.index_word) + 1, dim_model=dim_embeddings)
    
    num_classes = len(data[cols[model_name]].unique())
    
    return (X_train_tareas_pad, X_test_tareas_pad, 
            X_train_ocupacion_pad, X_test_ocupacion_pad,
            X_train_activ_principal_pad, X_test_activ_principal_pad,
            y_train_encoded, y_test_encoded, 
            tokenizer_tareas, tokenizer_ocupacion, tokenizer_activ_principal,
            matrix_embeddings_tareas, matrix_embeddings_ocupacion, matrix_embeddings_activ_principal,
            num_classes)

def train_model_1(tokenizer_tareas, tokenizer_ocupacion, tokenizer_activ_principal,
                  dim_embeddings, matrix_embeddings_tareas, matrix_embeddings_ocupacion, 
                  matrix_embeddings_activ_principal,
                  num_classes, X_train_tareas_pad, X_train_ocupacion_pad, X_train_activ_principal_pad,
                  y_train_encoded, model_name):
    
    # model hyperparameters
    parameters = {
        "nunits_lstm": 70,
        "batch_size": 64,
        "epochs": 100,
        "validation_split": 0.2
    }

    inputs = {
        "input_glosa1": Input(shape=(None,), dtype="int32", name="input_glosa1"),
        "input_glosa2": Input(shape=(None,), dtype="int32", name="input_glosa2"),
        "input_glosa3": Input(shape=(None,), dtype="int32", name="input_glosa3")
    }

    # Input y Embedding para la primera glosa
    x1 = Embedding(
            input_dim=len(tokenizer_tareas.index_word) + 1,
            output_dim=dim_embeddings,
            weights=[matrix_embeddings_tareas],
            trainable=False
        )(inputs["input_glosa1"])
    x1 = SpatialDropout1D(0.4)(x1)
    x1 = LSTM(parameters["nunits_lstm"], return_sequences=True)(x1)  
    x1 = LSTM(parameters["nunits_lstm"], return_sequences=False)(x1) 

    # Input y Embedding para la segunda glosa
    x2 = Embedding(
            input_dim=len(tokenizer_ocupacion.index_word) + 1,
            output_dim=dim_embeddings,
            weights=[matrix_embeddings_ocupacion],
            trainable=False
        )(inputs["input_glosa2"])
    x2 = SpatialDropout1D(0.4)(x2)
    x2 = LSTM(parameters["nunits_lstm"], return_sequences=True)(x2)  
    x2 = LSTM(parameters["nunits_lstm"], return_sequences=False)(x2)  

    # Input y Embedding para la tercera glosa
    x3 = Embedding(
            input_dim=len(tokenizer_activ_principal.index_word) + 1,
            output_dim=dim_embeddings,
            weights=[matrix_embeddings_activ_principal],
            trainable=False
        )(inputs["input_glosa3"])
    x3 = SpatialDropout1D(0.4)(x3)
    x3 = LSTM(parameters["nunits_lstm"], return_sequences=True)(x3) 
    x3 = LSTM(parameters["nunits_lstm"], return_sequences=False)(x3)  

    # Combinar las salidas
    combined = Concatenate()([x1, x2, x3])

    # Capa de salida
    outputs = Dense(num_classes, activation="softmax")(combined)
    model = Model(inputs=inputs, outputs=outputs)

    # compile model
    opt = Adam(learning_rate=1e-3)
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.summary()

    callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=3,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=10,
    )

    # Entrenar el modelo
    model.fit(
    {"input_glosa1": X_train_tareas_pad, "input_glosa2": X_train_ocupacion_pad, "input_glosa3": X_train_activ_principal_pad},
    y_train_encoded,
    epochs=parameters["epochs"],
    validation_split=parameters["validation_split"],
    batch_size=parameters["batch_size"],
    callbacks=[callback]
    )
    model.save(f"../models/modelo_1_{model_name}_lstm.h5")

if __name__ == "__main__":
    
    initialize_keras()

    # retrieve arguments passed from the calling script
    model_name = sys.argv[1]

    # load embeddings
    embeddings, dim_embeddings = load_embeddings()

    # generate artifacts
    X_train_tareas_pad, X_test_tareas_pad, X_train_ocupacion_pad, X_test_ocupacion_pad, X_train_activ_principal_pad, X_test_activ_principal_pad, y_train_encoded, y_test_encoded, tokenizer_tareas, tokenizer_ocupacion, tokenizer_activ_principal, matrix_embeddings_tareas, matrix_embeddings_ocupacion, matrix_embeddings_activ_principal, num_classes  = get_model_artifacts(model_name, embeddings, dim_embeddings)

    # train models
    train_model_1(tokenizer_tareas, tokenizer_ocupacion, tokenizer_activ_principal,
                  dim_embeddings, matrix_embeddings_tareas, matrix_embeddings_ocupacion, 
                  matrix_embeddings_activ_principal,
                  num_classes, X_train_tareas_pad, X_train_ocupacion_pad, X_train_activ_principal_pad,
                  y_train_encoded, model_name)