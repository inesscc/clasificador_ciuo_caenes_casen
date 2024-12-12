# -*- coding: utf-8 -*-

from utils.procesamiento import dir_input, dir_data_parquet
import pandas as pd

def read_input_data():
    '''
    Carga y consolida los datos en archivo parquet.
    Returns
    -------
    None.

    '''
    dir_data_parquet.mkdir(exist_ok=True)
    
    # Lee el archivo Stata
    casen = pd.read_stata(dir_input / 'Base de datos Casen 2022 STATA_18 marzo 2024.dta', convert_categoricals=False)
    glosas = pd.read_stata(dir_input / 'Casen2022_Ocod_bruto.dta')

    # Unir bases
    casen = casen[['folio', 'id_persona', 'oficio4_08', 'rama1_sub', 'rama4_sub', 'rama1','rama4']]
    df = pd.merge(glosas, casen, how='left', on=['folio', 'id_persona'])
    df_ciuo = df[['o9a', 'o9b', 'o23', 'o24', 'oficio4_08']]
    df_rama_paga = df[['o9a', 'o9b', 'o23', 'rama1_sub', 'rama4_sub']]
    df_rama = df[['o9a', 'o9b', 'o24', 'rama1', 'rama4']]

    # Eliminar NaN y No respuesta
    df_ciuo = df_ciuo[~df_ciuo['oficio4_08'].isin([-99, -88, -66])]
    df_ciuo = df_ciuo[df_ciuo['oficio4_08'].notna() & (df_ciuo['o9a'] != '') & (df_ciuo['o9b'] != '')]

    df_rama_paga = df_rama_paga[~df_rama_paga['rama4_sub'].isin([-99, -88, -66])]
    df_rama_paga = df_rama_paga[df_rama_paga['rama4_sub'].notna() & (df_rama_paga['o23'] != '')]

    df_rama = df_rama[~df_rama['rama4'].isin([-99, -88, -66])]
    df_rama = df_rama[df_rama['rama4'].notna() & (df_rama['o24'] != '')]

    # Crear dos dígitos
    df_ciuo['oficio4_08'] = df_ciuo['oficio4_08'].astype(int).astype(str)
    df_rama_paga['rama4_sub'] = df_rama_paga['rama4_sub'].astype(int).astype(str)
    df_rama['rama4'] = df_rama['rama4'].astype(int).astype(str)

    df_ciuo['largo'] = df_ciuo['oficio4_08'].str.len()
    df_rama_paga['largo'] = df_rama_paga['rama4_sub'].str.len()
    df_rama['largo'] = df_rama['rama4'].str.len()
        
    df_ciuo['oficio4_08'] = df_ciuo.apply(
    lambda row: f"0{row['oficio4_08']}" if row['largo'] == 3 else row['oficio4_08'],
    axis=1
        )
    
    df_rama_paga['rama4_sub'] = df_rama_paga.apply(
    lambda row: f"0{row['rama4_sub']}" if row['largo'] == 3 else row['rama4_sub'],
    axis=1
        )
    
    df_rama['rama4'] = df_rama.apply(
    lambda row: f"0{row['rama4']}" if row['largo'] == 3 else row['rama4'],
    axis=1
        )
        
    # Crear columnas a distintos dígitos
    df_ciuo['oficio2_08'] = df_ciuo['oficio4_08'].str[:2]
    df_rama_paga['rama2_sub'] = df_rama_paga['rama4_sub'].str[:2]
    df_rama['rama2'] = df_rama['rama4'].str[:2]
    
    # Seleccionar columnas
    df_ciuo = df_ciuo[['oficio2_08', 'oficio4_08', 'o9a', 'o9b', 'o23', 'o24']]
    df_rama_paga = df_rama_paga[['rama2_sub', 'rama4_sub', 'o23', 'o9a', 'o9b']]
    df_rama = df_rama[['rama2', 'rama4', 'o24', 'o9a', 'o9b']]

    # Renombrar columnas
    df_ciuo = df_ciuo.rename(columns={'o9a': 'glosa_ocupacion',
                                      'o9b': 'glosa_tareas',
                                      'o23': 'rama_paga',
                                      'o24': 'rama',
                                      'oficio4_08': 'ciuo_4d',
                                      'oficio2_08': 'ciuo_2d'})
    
    df_rama_paga = df_rama_paga.rename(columns={'o9a': 'glosa_ocupacion',
                                      'o9b': 'glosa_tareas',
                                      'o23': 'activ_principal',
                                      'rama4_sub': 'caenes_4d',
                                      'rama2_sub': 'caenes_2d'})
    
    df_rama = df_rama.rename(columns={'o9a': 'glosa_ocupacion',
                                      'o9b': 'glosa_tareas',
                                      'o24': 'activ_principal',
                                      'rama4': 'caenes_4d',
                                      'rama2': 'caenes_2d'})
    
    # Crear columna activ_principal en ciuo
    df_ciuo['activ_principal'] = df_ciuo['rama_paga'].where(df_ciuo['rama_paga'] != "", df_ciuo['rama'])
    df_ciuo = df_ciuo.drop(columns=['rama', 'rama_paga'])
    df_ciuo = df_ciuo.reset_index(drop=True)

    # Unir bases rama y rama_paga
    df_rama_compilada = pd.concat([df_rama, df_rama_paga], ignore_index=True)

    # Guardar en formato parquet
    df_rama_compilada.to_parquet(dir_data_parquet / 'data_consolidada_caenes.parquet')
    df_ciuo.to_parquet(dir_data_parquet / 'data_consolidada_ciuo.parquet')
    
    print('Data exportada exitosamente!')

if __name__ == "__main__":

    read_input_data()




