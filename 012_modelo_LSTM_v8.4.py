#!/usr/bin/env python
# coding: utf-8

# <table style="text-align: left; width: 100%;" border="0"
#  cellpadding="0" cellspacing="0">
# 
#   <tbody>
#     <tr align="center" style="height: 1px; background-color: rgb(0, 0, 0);">
#       <td><big><big><big><big>Grupo-7</big></big></big></td>
#       <td><big><big><big><big><span style="font-family: Calibri; color: white; font-weight: bold;">Laboratorio III</span></big></big></big></big></td>
#       <td><img src="https://i.imgur.com/YOQky86.png" title="source: imgur.com" style="width: 250px; height: auto;" /></td>
#     </tr>
#     <tr align="center">
#       <td colspan="3" rowspan="1"
#  style="height: 1px; background-color: rgb(68, 68, 100);"></td>
#     </tr>
#     <tr align="center">
#       <td colspan="3" rowspan="1"><big><big><big><big><span
#  style="font-family: Calibri;">Modelo LSTM <br>v8.4</span></big></big></big></big><br>
#       </td></tr>
#     <tr align="center">       
#       <td colspan="3" rowspan="1" style="height: 1px; background-color: rgb(68, 68, 100);"></td>
#     </tr>
#   </tbody>
# </table>
# 
# 
# <span style="font-family: Calibri; font-weight: bold; ">Autores:</span>
# <br style="font-family: Calibri; font-style: italic;">
# <span style="font-family: Calibri; font-style: italic;">
# - Aureliano Chavarria
# - Gastón Larregui
# - Patricia Nuñez
# </span>
# 
# <div class="footer">&copy; 2024</div>
# env py311 <br>
# jupyter nbconvert --to script 012_modelo_LSTM_v8.4.ipynb
# <br><br>
# <hr>
# <br><br>

# # Descripción del Script
# 
# Este script realiza una serie de pasos para preparar datos, entrenar y evaluar un modelo LSTM para predecir ventas. A continuación se describen las secciones principales del script y sus funciones:
# 
# 1. **Configuración Inicial**:
#    - Configura las opciones de pandas para una mejor visualización.
#    - Define funciones para registrar y calcular el tiempo de ejecución del script.
# 
# 2. **Importación de Bibliotecas y Carga de Datos**:
#    - Importa las bibliotecas necesarias.
#    - Define la ruta al archivo de datos y carga los datos en un DataFrame de pandas.
# 
# 3. **Preprocesamiento de Datos**:
#    - Realiza varias operaciones de preprocesamiento, como la eliminación de valores nulos, la creación de características adicionales, y el escalado de los datos.
# 
# 4. **Definición del Modelo LSTM**:
#    - Define la arquitectura del modelo LSTM utilizando Keras.
# 
# 5. **Entrenamiento y Validación**:
#    - Utiliza `TimeSeriesSplit` para realizar validación cruzada en series temporales.
#    - Entrena el modelo en cada división y evalúa su rendimiento.
# 
# 6. **Evaluación y Almacenamiento de Resultados**:
#    - Calcula métricas de error como MAE, RMSE, MAPE y TFE.
#    - Genera nombres de archivos basados en la fecha y hora actuales para guardar los resultados.
# 
# Este script está diseñado para realizar un análisis detallado y modelado predictivo de ventas utilizando LSTM. Desde la carga y preparación de datos hasta la evaluación del modelo, cada paso está cuidadosamente implementado para asegurar la precisión y efectividad del modelo predictivo.

# # Función para Calcular el Total Forecast Error
# 
# Este código define una función `calcular_total_forecast_error` que calcula el Total Forecast Error (TFE) a partir de dos series de datos: ventas reales y ventas pronosticadas. La función toma como parámetros dos series de pandas (`actual` y `forecast`) y devuelve el TFE, que es una métrica utilizada para evaluar la precisión de los pronósticos de ventas.
# 
# ## Ejemplo de Uso
# 
# El código incluye un ejemplo de uso en el que se crea un DataFrame con datos de ventas reales y pronosticadas, y se calcula el TFE para estos datos. El resultado se muestra en formato porcentual, proporcionando una medida clara de la precisión del pronóstico.

# In[1]:


import pandas as pd

def calcular_total_forecast_error(actual, forecast):
    """
    Calcula el Total Forecast Error dado un DataFrame con ventas reales y pronosticadas.

    Parámetros:
    actual (pd.Series): Serie con las ventas reales.
    forecast (pd.Series): Serie con las ventas pronosticadas.

    Retorna:
    float: El Total Forecast Error.
    """
    # Calcular el error absoluto
    abs_error = abs(actual - forecast)
    
    # Calcular el Total Forecast Error
    total_forecast_error = abs_error.sum() / actual.sum()
    
    print("\n\n-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print(f'    >>>>>>>>>>>>       Total Forecast Error: {total_forecast_error:.2%}     <<<<<<<<<<<<<<<<<<')
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------\n\n")

    
    return total_forecast_error

# Ejemplo de uso
# Crear un DataFrame de ejemplo
data = {
    'product_id': [1, 2, 3, 4, 5],
    'actual_sales': [100, 150, 200, 250, 300],
    'forecast_sales': [110, 145, 190, 260, 310]
}
df = pd.DataFrame(data)

# Calcular el Total Forecast Error
tfe = calcular_total_forecast_error(df['actual_sales'], df['forecast_sales'])
#print("\n\n-----------------------------------------------------------------------------")
#print("-----------------------------------------------------------------------------")
#print(f'    >>>>>>>>>>>>       Total Forecast Error: {tfe:.2%}     <<<<<<<<<<<<<<<<<<')
#print("-----------------------------------------------------------------------------")
#print("-----------------------------------------------------------------------------\n\n")



# # Funciones para Calcular el Tiempo de Ejecución
# 
# Este bloque de código define dos funciones para registrar y calcular el tiempo de ejecución de un script. Estas funciones son útiles para medir el rendimiento y la eficiencia del código.
# 
# ## Descripción de las Funciones
# 
# - `registrar_tiempo`: Esta función registra y devuelve el tiempo actual en un formato legible. Retorna una tupla que contiene el objeto `datetime` actual y su representación en cadena.
# 
# - `calcular_tiempo_transcurrido`: Esta función calcula el tiempo transcurrido entre dos instantes dados. Toma como parámetros dos objetos `datetime` (`inicio` y `fin`) y devuelve el tiempo transcurrido en segundos.
# 
# ## Ejemplo de Uso
# 
# El código incluye comentarios sobre cómo registrar el tiempo de inicio y finalización del script, así como cómo calcular y mostrar el tiempo transcurrido. Estos pasos ayudan a entender cuánto tiempo toma ejecutar el script completo, lo que es esencial para optimizar y mejorar el rendimiento del código.

# In[2]:


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Funciones para calcular el tiempo de ejecucion del script completo
#
# ---------------------------------------------------------------------------
import time
from datetime import datetime

def registrar_tiempo():
    """
    Registra y devuelve el tiempo actual en un formato legible.

    Retorna:
    tuple: Una tupla con el objeto datetime y su representación en cadena.
    """
    ahora = datetime.now()
    ahora_str = ahora.strftime("%Y-%m-%d %H:%M:%S")
    return ahora, ahora_str

def calcular_tiempo_transcurrido(inicio, fin):
    """
    Calcula el tiempo transcurrido entre dos instantes.

    Parámetros:
    inicio (datetime): El tiempo de inicio.
    fin (datetime): El tiempo de finalización.

    Retorna:
    float: El tiempo transcurrido en segundos.
    """
    return (fin - inicio).total_seconds()


# Registrar el tiempo de inicio al inicio del script
#inicio, inicio_str = registrar_tiempo()


# Registrar el tiempo de finalización al final del script
#fin, fin_str = registrar_tiempo()

# Calcular el tiempo transcurrido
#tiempo_transcurrido = calcular_tiempo_transcurrido(inicio, fin)

#print(f"Tiempo de inicio: {inicio_str}")
#print(f"Tiempo de finalización: {fin_str}")
#print(f"Tiempo transcurrido: {tiempo_transcurrido} segundos")


# <hr>
# 
# # Generación de Nombres de Archivos con Prefijo de Fecha y Hora
# 
# Funcion nombre_file(sfx)
# 
# Este bloque de código define una función para generar nombres de archivos CSV con un prefijo basado en la fecha y hora actuales, asegurando nombres únicos y organizados para los archivos de salida.
# 
# ## Descripción de la Función
# 
# - `nombre_file(sfx)`: Esta función genera un nombre de archivo único con un prefijo de fecha y hora en el formato `YYYYMMDD-HHMM-` seguido por un sufijo proporcionado (`sfx`). La función concatena este nombre con un directorio de salida predefinido.
# 
# ## Ejemplo de Uso
# 
# El código muestra cómo utilizar la función `nombre_file` para generar un nombre de archivo para una prueba específica (`suffix_name`). Además, incluye un ejemplo al final del script para calcular un error de pronóstico total (`tfe2`), añadirlo como sufijo al nombre del archivo y guardar las predicciones ajustadas en el archivo generado.
# 
# ## Guardar Parámetros en un Archivo JSON
# 
# El código también define una función adicional para guardar parámetros en un archivo JSON:
# 
# - `save_parameters(file_name, **params)`: Esta función guarda los parámetros proporcionados en un archivo JSON, facilitando el almacenamiento y recuperación de configuraciones de ejecución.
# 
# ## Ejemplo de Uso Completo
# 
# El código muestra cómo se genera el nombre del archivo con el prefijo de fecha y hora, cómo se incorpora el error de pronóstico total como parte del sufijo del archivo, y cómo se utilizan estas funciones para guardar las predicciones ajustadas y parámetros de configuración de manera organizada y estructurada.

# In[3]:


# --------------------------------------
# Funcion nombre_file(sfx)
#   - Genera el nombre del archivo .csv con prefijo datetime YYYY-MM-DD
from datetime import datetime
def nombre_file(sfx):
    # Obtener la fecha y hora actual en el formato requerido
    current_time = datetime.now().strftime('%Y%m%d-%H%M-')

    # Path to output dir
    output_dir = './666_Kaggle/Entregas/'    
    return(output_dir+current_time+sfx+'.csv')


# Indicar el nombre de la prueba
suffix_name = 'MongoAurelio' 

file_to_kaggle = nombre_file(suffix_name)
print(file_to_kaggle)


# ------------------------------------------------
# Esto va al final para escribir el archivo final

# Agrego el tfe2 al suffix del archivo
#   Calculo del total forecast error
#   tfe2 =  calcular_total_forecast_error(all_forecasts['y'], all_forecasts['yhat1'])
#    print(f'Total Forecast Error: {tfe2:.2%}')

tfe2 = 0.123456789
str_tfe2 = "_tfe2_" + str(round(tfe2, 4)) 

# Suffijo general para las dos salidas de archivos
#  Solo cambiar este valor
suffix_general = 'Sufijo-General'  + str_tfe2

# Usar la función nombre_file para asignar el nombre del archivo de salida para kaggle
suffix_to_kaagle_name = suffix_general
file_to_kaggle = nombre_file(suffix_to_kaagle_name)
# Colocar el nombre del df apropiado
#df_aGuardarEnDisco.to_csv(file_to_kaggle, index=False)

#all_forecasts.to_csv(file_to_kaggle+'all', index=False)
print(f'Predicciones ajustadas guardadas en {file_to_kaggle}')

# Fin
# ------------------------------------------------


# Función para guardar los parámetros en un file
import json
def save_parameters(file_name, **params):
    with open(file_name, 'w') as f:
        json.dump(params, f, indent=4)
# ------------------------------------------------


# ## Lecutura rapida de Dataset sell-z-780-all-LTSM.csv

# In[4]:


import pandas as pd

ventas_LTSM_path = './66_Datos/sell-z-780-all-LTSM.csv'
df_ventas = pd.read_csv(ventas_LTSM_path)

# Convertir la columna 'periodo' a tipo datetime
df_ventas['periodo'] = pd.to_datetime(df_ventas['periodo'], format='%Y-%m-%d')

# Formatear la fecha según el formato deseado
#print(df_ventas.info())

df_ventas.head(2)


# <hr>
# 
# # Configuración del Modelo LSTM
# 
# Este bloque de código describe la configuración y el entrenamiento de un modelo LSTM para predecir ventas. A continuación, se detallan los aspectos clave de la configuración del modelo:
# 
# ## Capas de la Red
# 
# - **Primera Capa LSTM**: Una capa LSTM con 50 unidades y `return_sequences=True` para permitir el paso de la secuencia completa a la siguiente capa.
# - **Segunda Capa LSTM**: Otra capa LSTM con 50 unidades y `return_sequences=False` para procesar la secuencia y producir una única salida.
# - **Capa Dropout**: Una capa Dropout con una tasa de 0.2 para prevenir el sobreajuste.
# - **Capa Densa**: Una capa densa con 25 unidades y activación ReLU.
# - **Capa de Salida**: Una capa densa con una sola unidad para producir la predicción final.
# 
# ## Parámetros
# 
# **Nota:** Estos son los parámetros por defecto de la función; sin embargo, pueden ser modificados en el momento de la llamada a la función.
# 
# - **Longitud de la Secuencia (seq_length)**: 12.
# - **Épocas (epochs)**: 100.
# - **Tamaño del Lote (batch_size)**: 32.
# - **Tasa de Aprendizaje (learning_rate)**: 0.001.
# - **Paciencia (patience)**: 20.
# - **Verbosidad (verbose)**: 0.
# - **Semilla (seed)**: 52.
# - **Estandarización (standard_scaler)**: Utiliza `StandardScaler` para escalar los datos si se selecciona (1).
# 
# ## Métodos de Normalización
# 
# - **Transformación Logarítmica**: Se aplica una transformación logarítmica a los datos de ventas (`tn`) para estabilizar la varianza y hacer que los datos se ajusten mejor al modelo.
# - **Estandarización**: Utiliza `StandardScaler` para escalar los datos si está habilitado, asegurando que los datos estén en el rango adecuado para el modelo LSTM.
# 
# ## Método para Completar los Datos Faltantes (NaN)
# 
# - **Reemplazo de Valores Cero**: Los valores de ventas cero se reemplazan por la media de las ventas no cero del mismo `product_id`.
# - **Reemplazo de NaN**: Los valores NaN en las ventas se reemplazan por la media global de ventas (`tn`).
# 
# ## Estrategia de Entrenamiento
# 
# - **División en Series Temporales**: Utiliza `TimeSeriesSplit` para dividir los datos en múltiples divisiones de entrenamiento y validación, asegurando que los datos más recientes se utilicen para validar el modelo.
# - **Pesos de las Muestras**: Calcula los pesos de las muestras basados en la proporción de las ventas totales, ajustando así la importancia de cada muestra durante el entrenamiento.
# 
# ## Callbacks Utilizados
# 
# - **Early Stopping**: Monitorea la pérdida de validación y detiene el entrenamiento si no hay mejora después de un número determinado de épocas (20).
# - **ReduceLROnPlateau**: Reduce la tasa de aprendizaje si la pérdida de validación no mejora, ayudando al modelo a encontrar el mínimo global.
# 
# ## Métricas de Evaluación
# 
# - **MAE (Error Absoluto Medio)**: Mide la diferencia promedio entre las predicciones y los valores reales.
# - **RMSE (Raíz del Error Cuadrático Medio)**: Mide la magnitud de los errores de predicción.
# - **MAPE (Error Porcentual Absoluto Medio)**: Mide el error de predicción en porcentaje.
# - **TFE (Error Total de Pronóstico)**: Mide el error absoluto total relativo a las ventas reales.
# 
# Este modelo LSTM está configurado para capturar patrones temporales en los datos de ventas, utilizando técnicas avanzadas de preprocesamiento y normalización para mejorar la precisión de las predicciones.

# In[5]:


import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import legacy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Verificar si hay GPUs disponibles
#print("GPUs disponibles: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Configurar el uso de la GPU
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        # Usar solo la primera GPU
#        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

#        # Limitar el crecimiento de la memoria de la GPU
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
        
#        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#    except RuntimeError as e:
#        print(e)

def calculate_weights(data):
    total = np.sum(data)
    weights = data / total
    return weights


def process_and_train_model(df_ventas, **kwargs):
    seq_length = kwargs.get('seq_length', 12)
    epochs = kwargs.get('epochs', 100)
    batch_size = kwargs.get('batch_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    patience = kwargs.get('patience', 20)
    verbose = kwargs.get('verbose', 0)
    seed = kwargs.get('seed', 52)
    standard_scaler = kwargs.get('standard_scaler', 1) # 1 StandardScaler yes, 0 StandardScaler No

    print(" **** process_and_train_model()  ***** ")
    print("   Cantidad de NaN en cat1: ", df_ventas['cat1'].isna().sum())
    print(" ********************************* ")

    
    
    # Preparación de datos
    df_full = df_ventas[['periodo', 'product_id', 'cat1', 'tn']].copy()
    
    # Identificar los valores cero antes de reemplazar
    valores_cero_antes = df_full[df_full['tn'] == 0].copy()

    # Calcular la media de 'tn' por 'product_id' y reemplazar los valores cero por esta media
    mean_tn_by_product = df_full[df_full['tn'] != 0].groupby('product_id')['tn'].mean()
    df_full.loc[df_full['tn'] == 0, 'tn'] = df_full['product_id'].map(mean_tn_by_product)

    # Reemplazar NaN con la media global de tn
    global_mean_tn = df_full['tn'].mean()
    df_full['tn'].fillna(global_mean_tn, inplace=True)

    # Aplicar la transformación logarítmica
    df_full['tn'] = np.log1p(df_full['tn'])
    
    # Estandarización de datos
    if standard_scaler == 1:
        scaler = StandardScaler()
        df_full['tn'] = scaler.fit_transform(df_full[['tn']])
    
    # Label Encoding para cat1
    label_encoder = LabelEncoder()
    df_full['cat1'] = label_encoder.fit_transform(df_full['cat1'])

    def create_sequences(data, seq_length):
        sequences = []
        labels = []
        for i in range(len(data) - seq_length):
            seq_data = data.iloc[i:i + seq_length].values
            sequences.append(seq_data)
            labels.append(data.iloc[i + seq_length]['tn'])
        return np.array(sequences), np.array(labels)

    product_sequences = {}
    weights_dict = {}
    for product_id in df_full['product_id'].unique():
        product_data = df_full[df_full['product_id'] == product_id].drop(columns=['product_id', 'periodo'])
        sequences, labels = create_sequences(product_data, seq_length)
        product_sequences[product_id] = (sequences, labels)
        
        # Calcular los pesos de las muestras
        weights = calculate_weights(product_data['tn'].values[:len(sequences)])
        weights_dict[product_id] = weights

    # Preparar los datos de entrenamiento
    X = np.concatenate([product_sequences[pid][0] for pid in product_sequences])
    y = np.concatenate([product_sequences[pid][1] for pid in product_sequences])
    product_ids = np.concatenate([[pid]*len(product_sequences[pid][1]) for pid in product_sequences])
    
    # Redimensionar los datos para que sean compatibles con LSTM
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Inicializar listas para almacenar las métricas de error
    mae_list = []
    rmse_list = []
    mape_list = []
    tfe_list = []

    # Utilizar TimeSeriesSplit para dividir los datos
    tscv = TimeSeriesSplit(n_splits=5)
    
    all_product_ids = []
    all_y_val = []
    all_y_pred_val = []

    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Obtener los pesos de las muestras para el conjunto de entrenamiento y validación
        sample_weights_train = np.concatenate([weights_dict[pid][:len(train_index)] for pid in product_sequences if pid in product_ids[train_index]])
        sample_weights_val = np.concatenate([weights_dict[pid][:len(val_index)] for pid in product_sequences if pid in product_ids[val_index]])

        # Verificar los tamaños antes de model.fit
        #print(f"X_train.shape: {X_train.shape}")
        #print(f"y_train.shape: {y_train.shape}")
        #print(f"sample_weights_train.shape: {sample_weights_train.shape}")
        #print(f"X_val.shape: {X_val.shape}")
        #print(f"y_val.shape: {y_val.shape}")
        #∫print(f"sample_weights_val.shape: {sample_weights_val.shape}")

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, X.shape[2])),  # Primera capa LSTM
            LSTM(50, return_sequences=False),  # Segunda capa LSTM
            Dropout(0.2),  # Mantener el dropout para regularización
            Dense(25, activation='relu'),  # Reducir el tamaño de la capa densa
            Dense(1)  # Capa de salida
        ])
        
        optimizer = legacy.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', weighted_metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)

        history = model.fit(X_train, y_train, sample_weight=sample_weights_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val, sample_weights_val), callbacks=[early_stopping, reduce_lr], verbose=verbose)

        # Generar predicciones para el split actual
        y_pred_val = model.predict(X_val)

        y_val = y_val.reshape(-1, 1)
        y_pred_val = y_pred_val.reshape(-1, 1)
        
        # Invertir normalizacion StandarScarler si esta fue seleccionada.
        if standard_scaler == 1:
            y_pred_val = scaler.inverse_transform(y_pred_val)
            y_val = scaler.inverse_transform(y_val)

        # Invertir la transformación logarítmica para los valores de validación y predicción
        y_pred_val = np.expm1(y_pred_val)
        y_val = np.expm1(y_val)

        # Calcular las métricas de error para el split actual
        mae = mean_absolute_error(y_val, y_pred_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        mape = np.mean(np.abs((y_val - y_pred_val) / y_val)) * 100
        tfe = np.sum(np.abs(y_val - y_pred_val)) / np.sum(np.abs(y_val))

        # Almacenar las métricas en las listas correspondientes
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        tfe_list.append(tfe)

        all_product_ids.extend(product_ids[val_index])
        all_y_val.extend(y_val)
        all_y_pred_val.extend(y_pred_val)

    
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    avg_mape = np.mean(mape_list)
    avg_tfe = np.mean(tfe_list)

    # Generar predicciones finales con el modelo entrenado
    X_pred = np.array([product_sequences[pid][0][-1] for pid in product_sequences])
    predictions = model.predict(X_pred)
    
    # Invertir normalizacion StandarScarler si esta fue seleccionada.
    if standard_scaler == 1:
        predictions = scaler.inverse_transform(predictions)
    
    # Invertir la transformación logarítmica
    predictions = np.expm1(predictions)

    final_predictions = pd.DataFrame({
        'product_id': df_full['product_id'].unique(),
        'tn_pred': predictions.flatten()
    })

    final_product_ids = np.array(all_product_ids)
    final_y_val = np.array(all_y_val)
    final_y_pred_val = np.array(all_y_pred_val)

    # Asegurarse de que todos los productos están presentes en all_results
    all_results = pd.DataFrame({
        'product_id': final_product_ids,
        'y_val': final_y_val.flatten(),
        'y_pred_val': final_y_pred_val.flatten()
    })

    # Devolver las métricas promedio junto con las predicciones y resultados de validación
    metrics = {
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'avg_mape': avg_mape,
        'avg_tfe': avg_tfe
    }
    
    return final_predictions, all_results, weights_dict, metrics


# <hr>
# 
# # Descripción de la Función `master_of_the_universe()`
# 
# Esta función se encarga de realizar predicciones de ventas utilizando un modelo LSTM y gestionar las salidas y métricas de rendimiento para los períodos de enero y febrero de 2020. A continuación se detalla lo que realiza cada parte de la función:
# 
# ## Predicción de Ventas para Enero 2020
# 
# - **Llamada a la Función de Entrenamiento**: Utiliza la función `process_and_train_model` con un conjunto de parámetros para entrenar el modelo y obtener las predicciones.
# - **Almacenamiento de Resultados**: Renombra y guarda las predicciones en un archivo CSV.
# - **Cálculo de Métricas**: Calcula las métricas de error (MAE, RMSE, MAPE, TFE) para las predicciones.
# 
# ## Preparación de Datos para Febrero 2020
# 
# - **Actualización del Período**: Calcula el siguiente período y actualiza las ventas para este período con las predicciones obtenidas.
# - **Manejo de Datos Faltantes**: Completa los valores faltantes (`NaN`) en la categoría `cat1` utilizando un mapeo basado en `product_id`.
# 
# ## Predicción de Ventas para Febrero 2020
# 
# - **Entrenamiento del Modelo con Datos Actualizados**: Reentrena el modelo LSTM utilizando los datos actualizados hasta el último período.
# - **Almacenamiento de Resultados**: Guarda las nuevas predicciones en un archivo CSV.
# - **Verificación de Valores Infinitos o NaN**: Reemplaza los valores infinitos y faltantes con valores finitos o la media, respectivamente.
# - **Cálculo de Métricas**: Calcula las métricas de error para las predicciones de noviembre y diciembre de 2019, y combina estas métricas con los parámetros utilizados.
# 
# ## Guardar Métricas y Parámetros
# 
# - **Almacenamiento en JSON**: Guarda las métricas de rendimiento y los parámetros de configuración en un archivo JSON para su posterior análisis y referencia.
# 
# Esta función gestiona el proceso completo de predicción y evaluación, asegurando que los resultados se guarden de manera organizada y que las métricas de rendimiento se calculen y almacenen adecuadamente.

# In[6]:


def master_of_the_universe():
    # --------------------------------------------------------------------------
    #  Salida Prediccion Enero 2020
    # --------------------------------------------------------------------------

    # Llama a la función con el diccionario de parámetros
    results, df_errors_sorted, weights_dict, metrics = process_and_train_model(df_ventas, **parameters)

    # Ahora guarda los resultados como de costumbre
    results_toCsv = results.copy()
    results_toCsv = results_toCsv.rename(columns={'tn_pred': 'tn'})

    # Guardar el archivo final
    #suffix_general = 'LSTM_v8v4_ene' + "_tfe2_" + str(round(metrics['avg_tfe'], 4))
    #file_to_kaggle = nombre_file(suffix_general)
    #results_toCsv.to_csv(file_to_kaggle, index=False)

    # Calcula las métricas para noviembre y diciembre 2019
    #mae_nov_dec_2019 = mean_absolute_error(df_errors_sorted['y_val'], df_errors_sorted['y_pred_val'])
    #rmse_nov_dec_2019 = np.sqrt(mean_squared_error(df_errors_sorted['y_val'], df_errors_sorted['y_pred_val']))
    #mape_nov_dec_2019 = np.mean(np.abs((df_errors_sorted['y_val'] - df_errors_sorted['y_pred_val']) / df_errors_sorted['y_val'])) * 100
    #tfe_nov_dec_2019 = calcular_total_forecast_error(df_errors_sorted['y_val'], df_errors_sorted['y_pred_val'])

    # Combina las métricas con los parámetros
    #metrics_combined = {
    #    "validation": {
    #        "avg_mae": metrics["avg_mae"],
    #        "avg_rmse": metrics["avg_rmse"],
    #        "avg_mape": metrics["avg_mape"],
    #        "avg_tfe": metrics["avg_tfe"]
    #    },
    #    "future_predictions": {
    #        "mae_nov_dec_2019": mae_nov_dec_2019,
    #        "rmse_nov_dec_2019": rmse_nov_dec_2019,
    #        "mape_nov_dec_2019": mape_nov_dec_2019,
    #        "tfe_nov_dec_2019": tfe_nov_dec_2019
    #    }
    #}
    #metrics_combined.update(parameters)

    # Guardar las métricas y parámetros en un archivo JSON
    #metrics_file = file_to_kaggle + ".metrics.json"
    #save_parameters(metrics_file, **metrics_combined)

    #print(f'Predicciones ajustadas guardadas en {file_to_kaggle}')
    #print(f'Métricas y parámetros guardados en {metrics_file}')

    # --------------------------------------------------------------------------
    #  Salida Prediccion Febrero 2020
    # --------------------------------------------------------------------------

    # Calcular el siguiente período (suponiendo que los periodos son mensuales)
    df_ventas['periodo'] = pd.to_datetime(df_ventas['periodo'], format='%Y%m')
    ultimo_periodo = df_ventas['periodo'].max()
    siguiente_periodo = ultimo_periodo + pd.DateOffset(months=1)

    # Actualizar las 'tn' del siguiente período con las 'tn' de results
    df_ventas['periodo'] = df_ventas['periodo'].dt.strftime('%Y%m')
    results_toCsv['periodo'] = siguiente_periodo.strftime('%Y%m')
    df_ventas2 = pd.concat([df_ventas, results_toCsv], ignore_index=True)

    # Vuelvo a convertir a 'periodo' a tipo datetime
    df_ventas['periodo'] = pd.to_datetime(df_ventas['periodo'], format='%Y%m')
    df_ventas2['periodo'] = pd.to_datetime(df_ventas2['periodo'], format='%Y%m')
    df_ventas['periodo'] = df_ventas['periodo'].dt.strftime('%Y-%m-%d')
    df_ventas2['periodo'] = df_ventas2['periodo'].dt.strftime('%Y-%m-%d')

    # Para el periodo agregado a df_ventas2 los valores para cat1 (y otros atributos) son completados con NaN.
    # Las siguientes lineas reemplazan en cat1 los valores NaN del periodo agregado a df_ventas con su código correspondiente.
    df_referencia = df_ventas2[['product_id', 'cat1']].dropna().drop_duplicates()
    cat1_mapping = df_referencia.set_index('product_id')['cat1'].to_dict()
    df_ventas2['cat1'] = df_ventas2['cat1'].fillna(df_ventas2['product_id'].map(cat1_mapping))

    print(" **** Master of the Universe ***** ")
    print("   Cantidad de NaN en cat1: ", df_ventas2['cat1'].isna().sum())
    print(" ********************************* ")

    # Entrenar el modelo con datos hasta el último período actualizado
    results2, df_errors_sorted2, weights_dict2, metrics2 = process_and_train_model(df_ventas2, **parameters)
    results2 = results2.rename(columns=({'tn_pred': 'tn'}))

    # Guardar el archivo final
    suffix_general = 'LSTM_v8v4_feb' + "_tfe2_" + str(round(metrics2['avg_tfe'], 4))
    file_to_kaggle = nombre_file(suffix_general)
    results2.to_csv(file_to_kaggle, index=False)





    # Verificar si hay valores infinitos o NaN
    #  esto se realiza para evitar un error en la ejecucion del codigo
    print(" -----------------------------------------------")
    print("  Verificando valores infinitos")
    print(np.isinf(df_errors_sorted2['y_pred_val']).sum())
    print(np.isnan(df_errors_sorted2['y_pred_val']).sum())
    print(np.isinf(df_errors_sorted2['y_val']).sum())
    print(np.isnan(df_errors_sorted2['y_val']).sum())
    

    # Reemplaza infinitos con valores grandes finitos
    df_errors_sorted2['y_pred_val'] = np.where(np.isinf(df_errors_sorted2['y_pred_val']), np.finfo(np.float32).max, df_errors_sorted2['y_pred_val'])
    df_errors_sorted2['y_val'] = np.where(np.isinf(df_errors_sorted2['y_val']), np.finfo(np.float32).max, df_errors_sorted2['y_val'])

    # Reemplaza NaN con el promedio o con 0, dependiendo del caso
    df_errors_sorted2['y_pred_val'].fillna(df_errors_sorted2['y_pred_val'].mean(), inplace=True)
    df_errors_sorted2['y_val'].fillna(df_errors_sorted2['y_val'].mean(), inplace=True)
    
    
    
    
    # Calcula las métricas para noviembre y diciembre 2019
    mae_nov_dec_2019_2 = mean_absolute_error(df_errors_sorted2['y_val'], df_errors_sorted2['y_pred_val'])
    rmse_nov_dec_2019_2 = np.sqrt(mean_squared_error(df_errors_sorted2['y_val'], df_errors_sorted2['y_pred_val']))
    mape_nov_dec_2019_2 = np.mean(np.abs((df_errors_sorted2['y_val'] - df_errors_sorted2['y_pred_val']) / df_errors_sorted2['y_val'])) * 100
    tfe_nov_dec_2019_2 = calcular_total_forecast_error(df_errors_sorted2['y_val'], df_errors_sorted2['y_pred_val'])

    # Combina las métricas con los parámetros
    metrics_combined_2 = {
        "validation": {
            "avg_mae": metrics2["avg_mae"],
            "avg_rmse": metrics2["avg_rmse"],
            "avg_mape": metrics2["avg_mape"],
            "avg_tfe": metrics2["avg_tfe"]
        },
        "future_predictions": {
            "mae_nov_dec_2019": mae_nov_dec_2019_2,
            "rmse_nov_dec_2019": rmse_nov_dec_2019_2,
            "mape_nov_dec_2019": mape_nov_dec_2019_2,
            "tfe_nov_dec_2019": tfe_nov_dec_2019_2
        }
    }
    metrics_combined_2.update(parameters)

    # Guardar las métricas y parámetros en un archivo JSON
    metrics_file_2 = file_to_kaggle + ".metrics.json"
    save_parameters(metrics_file_2, **metrics_combined_2)

    print(f'Predicciones ajustadas guardadas en {file_to_kaggle}')
    print(f'Métricas y parámetros guardados en {metrics_file_2}')

    return df_ventas2, weights_dict2


# <hr>
# 
# # Configuración y Ejecución de Combinaciones de Parámetros para el Modelo LSTM
# 
# Este bloque de código define una serie de parámetros y combina sus valores para entrenar y evaluar un modelo LSTM para predicción de ventas, controlando si se incluyen predicciones futuras o no.
# 
# ## Control de Predicciones Futuras
# 
# - **Variable `future_prediction`**: Controla si se incluyen las predicciones para febrero 2020 (`True`) o si el entrenamiento se limita hasta octubre 2019 (`False`).
# 
# ## Parámetros Definidos
# 
# - **SEEDs**: Lista de semillas para reproducibilidad.
# - **seq_lengths**: Lista de longitudes de secuencia para el modelo LSTM.
# - **epochs_list**: Lista de cantidades de épocas para entrenar el modelo.
# - **batch_sizes**: Lista de tamaños de lote.
# - **learning_rates**: Lista de tasas de aprendizaje.
# - **patience_list**: Lista de valores de paciencia para el early stopping.
# - **verbose_list**: Lista de valores para el nivel de verbosidad del entrenamiento.
# - **standard_scalerS**: Lista de valores para indicar si se utiliza `StandardScaler` (1) o no (0).
# 
# ## Cálculo del Número Total de Combinaciones
# 
# El código calcula el número total de combinaciones posibles de parámetros multiplicando las longitudes de las listas definidas para cada parámetro.
# 
# ## Iteración sobre las Combinaciones de Parámetros
# 
# - **Fijación de la Semilla**: Para reproducibilidad, se fija la semilla para numpy, tensorflow y random en cada iteración.
# - **Carga y Preprocesamiento de Datos**: Carga los datos de ventas desde un archivo CSV y convierte la columna `periodo` a tipo datetime.
# - **Filtrado de Datos**: Si `future_prediction` es `False`, filtra los datos para incluir solo hasta octubre 2019.
# - **Definición de Parámetros**: Define un diccionario `parameters` con los valores actuales de los parámetros de la iteración.
# - **Registro de Tiempos**: Registra el tiempo de inicio y finalización de cada iteración para calcular el tiempo transcurrido.
# - **Impresión de Parámetros y Tiempo de Ejecución**: Imprime los parámetros utilizados y el tiempo de ejecución de la iteración.
# 
# ## Llamada a la Función Principal
# 
# Llama a la función `master_of_the_universe` con los parámetros definidos para entrenar y evaluar el modelo LSTM, y guardar los resultados y métricas.
# 
# ## Resumen
# 
# Este bloque de código asegura que todas las combinaciones posibles de parámetros sean evaluadas, registrando tiempos de ejecución y resultados para cada una, lo que permite una comparación exhaustiva y detallada del rendimiento del modelo LSTM bajo diferentes configuraciones.

# In[7]:


import pandas as pd
import numpy as np
import random
import tensorflow as tf

# Definir la variable para controlar si se incluyen las predicciones futuras o no
future_prediction = False  # True para predecir febrero 2020, False para entrenar solo hasta octubre 2019

# Fijar la semilla para reproducibilidad
SEEDs = [52]
seq_lengths = [3,6,12]
epochs_list = [50,20,100]
batch_sizes = [32,64,16]
learning_rates = [0.0001]
patience_list = [10]
verbose_list = [0]
standard_scalerS = [1,0] # [0, 1]



# Calcular el número total de combinaciones:
total_combinations = len(SEEDs) * len(seq_lengths) * len(epochs_list) * len(batch_sizes) * \
                     len(learning_rates) * len(patience_list) * len(verbose_list) * len(standard_scalerS)

print("Total de combinaciones:", total_combinations)


# Registrar el tiempo de inicio al inicio del script
inicio, inicio_str = registrar_tiempo()


# Iterar sobre todas las combinaciones de valores de parámetros
for seed in SEEDs:
    # Fijar la semilla para reproducibilidad    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    for standard_scaler in standard_scalerS:    
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for patience in patience_list:
                        for verbose in verbose_list:
                            for seq_length in seq_lengths:                                                            
                                # Cargar los datos
                                ventas_LTSM_path = './66_Datos/sell-z-780-all-LTSM.csv'
                                df_ventas = pd.read_csv(ventas_LTSM_path)

                                # Convertir la columna 'periodo' a tipo datetime
                                df_ventas['periodo'] = pd.to_datetime(df_ventas['periodo'], format='%Y-%m-%d')
                                
                                # Filtrar los datos si future_prediction es False hasta octubre 2019
                                if not future_prediction:
                                    df_ventas = df_ventas[df_ventas['periodo'] <= '2019-10-01']
                                    
                                # Definir los parámetros utilizados
                                parameters = {
                                    "seq_length": seq_length,
                                    "epochs": epochs,
                                    "batch_size": batch_size,
                                    "learning_rate": learning_rate,
                                    "patience": patience,
                                    "verbose": verbose,
                                    "seed": seed,
                                    "standard_scaler": standard_scaler,
                                    "future_predicion":future_prediction
                                }
                                # Registrar el tiempo de finalización al final del script
                                fin, fin_str = registrar_tiempo()
                                # Calcular el tiempo transcurrido
                                tiempo_transcurrido = calcular_tiempo_transcurrido(inicio, fin)

                                print('\n')
                                print('# -------------------------------------------------------------------------------------------------------------------')
                                print('# Tiempo de Ejecucion:')
                                print(f"#    Tiempo de inicio: {inicio_str}")
                                print(f"#    Tiempo de actual: {fin_str}")
                                print('# ') 
                                print(f"#    Tiempo transcurrido: {tiempo_transcurrido} segundos")             
                                print('# ')
                                print('# LSTM Parameters:')
                                print(f'#   seed={seed}, seq_length={seq_length}, epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, patience={patience}, verbose={verbose}, standard_scaler={standard_scaler} ')
                                print('# -------------------------------------------------------------------------------------------------------------------')
                                print('\n')
                                
                                # Llamar a la función principal con los parámetros
                                df_ventas2, weights_dict = master_of_the_universe()

                                # Registrar el tiempo de finalización al final del script
                                #fin, fin_str = registrar_tiempo()

                                # Calcular el tiempo transcurrido
                                #tiempo_transcurrido = calcular_tiempo_transcurrido(inicio, fin)

                                #print(f"Tiempo de inicio: {inicio_str}")
                                #print(f"Tiempo de finalización: {fin_str}")
                                #print(f"Tiempo transcurrido: {tiempo_transcurrido} segundos")


print('\n')
print('# -------------------------------------------------------------------------------------------------------------------')
print('# Tiempo de Ejecucion:')
print(f"#    Tiempo de inicio: {inicio_str}")
print(f"#    Tiempo de actual: {fin_str}")
print('# ') 
print(f"#    Tiempo transcurrido: {tiempo_transcurrido} segundos")             
print('# ')
print('# LSTM Parameters:')
print(f'#   seed={seed}, seq_length={seq_length}, epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, patience={patience}, verbose={verbose}, standard_scaler={standard_scaler} ')
print('# -------------------------------------------------------------------------------------------------------------------')
print('\n')


# In[ ]:




