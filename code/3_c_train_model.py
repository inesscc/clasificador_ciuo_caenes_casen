import pandas as pd
import sys
from transformers import AutoTokenizer, BertForSequenceClassification
import numpy as np
import evaluate
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from utils.model import initialize_keras, split_data, tokenize_data, create_matrix_embeddings, save_pkl, load_embeddings, read_data_models

metric = evaluate.load("accuracy") 

def tokenize_function(examples, tokenizer):
    return tokenizer(examples.tolist(), padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model_sabe(model_name):

    data = read_data_models(model_name)
    
    X_train, X_test, y_train, y_test = split_data(data, 'CIUO_4d', 0.3 )

    X_train['glosa'] = X_train['glosa_ocupacion'] + ' [SEP] ' + X_train['glosa_tareas'] + ' [SEP] ' + X_train['activ_principal']
    X_test['glosa'] = X_test['glosa_ocupacion'] + ' [SEP] ' + X_test['glosa_tareas'] + ' [SEP] ' + X_test['activ_principal']

    num_clases_unicas = len(np.unique(y_train))
    print(num_clases_unicas)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test) 

    # Cargar modelo SABE
    model_sabe = BertForSequenceClassification.from_pretrained("SABE-SENCE/SABE_ENCUESTAS", num_labels=num_clases_unicas, ignore_mismatched_sizes=True)

    # Cargar tokenizer SABE
    tokenizer_sabe = AutoTokenizer.from_pretrained("SABE-SENCE/SABE_ENCUESTAS")

    # Tokenizar    
    tokenized_train = tokenize_function(X_train['glosa'], tokenizer_sabe)
    tokenized_test = tokenize_function(X_test['glosa'], tokenizer_sabe)

    tokenized_train = dict(tokenized_train)
    tokenized_test = dict(tokenized_test)

    tokenized_train["label"] = y_train_encoded
    tokenized_test["label"] = y_test_encoded

    # Convierte los datos tokenizados en objetos de Dataset
    train_dataset = Dataset.from_dict(tokenized_train)
    test_dataset = Dataset.from_dict(tokenized_test)

    # Argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="test_trainer",         # Directorio de salida
        num_train_epochs=2,                # Número de épocas
        per_device_train_batch_size=8,     # Tamaño de batch para entrenamiento
        per_device_eval_batch_size=8,      # Tamaño de batch para evaluación
        eval_strategy="epoch",       # Evaluar cada época
    )

    trainer = Trainer(
    model=model_sabe,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    )

    print("Comienzo del entrenamiento...")

    trainer.train(resume_from_checkpoint=True)

    print("Entrenamiento completo.")

    trainer.save_model("../models/modelo_sabe_fine_tunning")
    tokenizer_sabe.save_pretrained("../models/modelo_sabe_fine_tunning")

if __name__ == "__main__":
    
    # Inicializar Keras
    initialize_keras()

    # Obtener el argumento 'model_name' desde la línea de comandos
    model_name = sys.argv[1]

    # Entrenar el modelo
    train_model_sabe(model_name)

    
    