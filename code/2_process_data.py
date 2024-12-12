# -*- coding: utf-8 -*-

import pandas as pd
from utils.procesamiento import dir_data_parquet, dir_data_proc
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import string

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

def process_data():
    '''
    Procesa datos de texto y exporta data procesada.

    Returns
    -------
    None.

    '''
    df_ciuo = pd.read_parquet(dir_data_parquet / 'data_consolidada_ciuo.parquet')
    df_rama = pd.read_parquet(dir_data_parquet / 'data_consolidada_caenes.parquet')

    df_ciuo['glosa_ocupacion'] = df_ciuo['glosa_ocupacion'].map(lambda x: process_glosa(x))
    df_ciuo['glosa_tareas'] = df_ciuo['glosa_tareas'].map(lambda x: process_glosa(x))
    df_ciuo['activ_principal'] = df_ciuo['activ_principal'].map(lambda x: process_glosa(x))

    df_rama['glosa_ocupacion'] = df_rama['glosa_ocupacion'].map(lambda x: process_glosa(x))
    df_rama['glosa_tareas'] = df_rama['glosa_tareas'].map(lambda x: process_glosa(x))
    df_rama['activ_principal'] = df_rama['activ_principal'].map(lambda x: process_glosa(x))

    df_ciuo.to_parquet(dir_data_proc / "data_procesada_ciuo.parquet")
    df_rama.to_parquet(dir_data_proc / "data_procesada_caenes.parquet")

    print('Datos procesados exportados exitosamente!')

if __name__ == "__main__":
    process_data()