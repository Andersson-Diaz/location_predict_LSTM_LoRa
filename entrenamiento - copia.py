#librerias necesarias
from tokenize import Number
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout

import time
from datetime import datetime
from datetime import timedelta
import math

#datos para la conexion a la base de datos
hostname = '82.180.175.58'
username = 'u813407238_lora'
password = 'Seguimiento_lora_123'
database = 'u813407238_seguimiento'

import MySQLdb
# inicialmente hace la conexion con la base de datos
myConnection = MySQLdb.connect( host=hostname, user=username, passwd=password, db=database )
import pandas as pd
# genera la lectura de la base de datos
dataset= pd.read_sql("SELECT * FROM LoRaWAN_messages_calle_5 order by id",myConnection)
print("Va a imprimir el dataset leido de la BD...")
dataset.drop(index=dataset[dataset['latitude']=='0'].index, inplace=True)
dataset.info()
time = dataset['hour']
dataset['latitude']=dataset['latitude'].astype('float64')
dataset['longitude']=dataset['longitude'].astype('float64')
time_step = 30
last = int(len(dataset)/5.0)
set_entrenamiento = dataset[:-last]
set_validacion = dataset[-last-time_step:]
set_entrenamiento.reset_index(inplace=True, drop=True)
set_validacion.reset_index(inplace=True, drop=True)
x= np.column_stack((set_entrenamiento.iloc[:,[4]],set_entrenamiento.iloc[:,[5]],set_entrenamiento.iloc[:,[8]],set_entrenamiento.iloc[:,[12]]))
# Normalización del set de entrenamiento
sc1 = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc1.fit_transform(x)

#Declaracion de vectores de entrada y salida para el entrenamiento
X_train = []
Y_train = []

m = len(set_entrenamiento_escalado)
#Crea las ventanas de datos
for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0:5])
    # Y: el siguiente dato despues de la ventana de datos
    Y_train.append(set_entrenamiento_escalado[i,0:5])

#Transforma las listas en vectores
X_train, Y_train = np.array(X_train), np.array(Y_train)

#Reforma el vector para que se ajuste al modelo en keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))

# Red LSTM
#Para crear la red debemos primero definir el tamaño de los datos de entrada y del dato de salida,
#así como el número total de neuronas (100):
dim_entrada = (X_train.shape[1],4)
dim_salida = 4
na = 200

from keras.layers import Bidirectional

modelo = Sequential()
modelo.add(LSTM(units=na,  return_sequences=True, input_shape=dim_entrada))
modelo.add(LSTM(200,  return_sequences=True))
modelo.add(LSTM(200,  return_sequences=True))
modelo.add(LSTM(200,  input_shape=dim_entrada))
modelo.add(Dropout(0.2))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='Adam', loss='mse')
modelo.fit(X_train,Y_train,epochs=500,batch_size=32)
print(modelo.summary())
modelo.save('path_to_my_model.h5')
import joblib
joblib.dump(sc1, 'scaler.save')

#termina el entrenamiento del modelo para la predicción de la ubicacion
#Comienza el entrenamiento del modelo para la prediccion del tiempo

#Definicion de funcion que retorna la distancia entre puntos geográficos
def haversine(lat1, lon1, lat2, lon2):
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia=2*R*math.asin(math.sqrt(a))
    return distancia

#Crea una lista con la distancia entre un punto de ubicacion y el punto anterior desde el set de entrenamiento
distancia_entrenamiento = []
distancia_entrenamiento.append(0)
for i in range(0, len(set_entrenamiento)-1):
    distancia_entrenamiento.append(haversine(set_entrenamiento.iat[i,4],set_entrenamiento.iat[i,5],set_entrenamiento.iat[i+1,4],set_entrenamiento.iat[i+1,5]))

#Se crea un dataframe con la lista que contiene la distancia del conjunto de entrenamiento
d_d_e = pd.DataFrame(distancia_entrenamiento, columns=['distancia'])

#Crea una lista con la distancia entre un punto de ubicacion y el punto anterior desde el set de validacion
distancia_validacion = []
distancia_validacion.append(0)
for i in range(0, len(set_validacion)-1):
    distancia_validacion.append(haversine(set_validacion.iat[i,4],set_validacion.iat[i,5],set_validacion.iat[i+1,4],set_validacion.iat[i+1,5]))

#Se crea un dataframe con la lista que contiene la distancia del conjunto de validacion
d_d_v = pd.DataFrame(distancia_validacion, columns=['distancia'])

#toma la hora del conjunto de datos de entrenamiento
time_entrenamiento = set_entrenamiento['hour']
#toma la hora del conjunto de datos de validacion
time_validacion = set_validacion['hour']


#Calcula la diferencia de tiempo entre un punto de ubicacion y el anterior en el set de entrenamiento
medida_de_tiempo_entrenamiento = []
for i in range(0,len(time_entrenamiento)-1):
    medida_de_tiempo_entrenamiento.append(time_entrenamiento[i+1]-time_entrenamiento[i])
medida_de_tiempo_entrenamiento

#Calcula la diferencia de tiempo entre un punto de ubicacion y el anterior en el set de validacion
medida_de_tiempo_validacion = []
for i in range(0,len(time_validacion)-1):
    medida_de_tiempo_validacion.append(time_validacion[i+1]-time_validacion[i])
medida_de_tiempo_validacion

#Creamos un objeto deltatime de valor 1 segundo
#Al dividir deltatime / deltatime se obtiene un valor de tipo float
#Al dividir sobre un segundo se obtiene un valor de tiempo en segundos
delta = timedelta(
    days=0,
    seconds=1,
    microseconds=0,
    milliseconds=0,
    minutes=0,
    hours=0,
    weeks=0 )

#Para EL ENTRENAMIENTO Cambiamos el tipo de dato a flotante, el resultado es un número en segundos
duracion_entrenamiento = []
duracion_entrenamiento.append(10)
for i in range(0,len(medida_de_tiempo_entrenamiento)):
    duracion_entrenamiento.append(medida_de_tiempo_entrenamiento[i]/delta)
duracion_entrenamiento

#Agrupamos tiempo y distancia para usarlas como entradas del modelo de entrenamiento
n_e= np.column_stack((duracion_entrenamiento,distancia_entrenamiento))

#Para la validacion Cambiamos el tipo de dato a flotante, el resultado es un número en segundos
duracion_validacion = []
duracion_validacion.append(10)
for i in range(0,len(medida_de_tiempo_validacion)):
    duracion_validacion.append(medida_de_tiempo_validacion[i]/delta)
duracion_validacion

#Agrupamos tiempo y distancia para usarlas como entradas en la prediccion del modelo
n_v= np.column_stack((duracion_validacion,distancia_validacion))

#Se crea un dataframe a partir de la lista
duracion_data_entrenamiento = pd.DataFrame(duracion_entrenamiento, columns=['duracion'])
duracion_data_entrenamiento

#Se crea un dataframe a partir de la lista
duracion_data_validacion = pd.DataFrame(duracion_validacion, columns=['duracion'])
duracion_data_validacion

#normalizamos el conjunto de datos de entrada
sc = MinMaxScaler(feature_range=(0,1))
time_entrenamiento_escalado = sc.fit_transform(n_e)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción se hace a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step_t = 30
X_train_duracion = []
Y_train_duracion = []
n = len(n_e)

#Se agregan las ventanas del set de entrenamiento
for i in range(time_step,n):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train_duracion.append(time_entrenamiento_escalado[i-time_step:i,0:2])
    # Y: el siguiente dato
    Y_train_duracion.append(time_entrenamiento_escalado[i,0:2])

#Se pasan las listas a arrays
X_train_duracion, Y_train_duracion = np.array(X_train_duracion), np.array(Y_train_duracion)

# Reshape X_train para que se ajuste al modelo en Keras
X_train_duracion = np.reshape(X_train_duracion, (X_train_duracion.shape[0], X_train_duracion.shape[1], 2))

# Red LSTM
#Para crear la red debemos primero definir el tamaño de los datos de entrada y del dato de salida,
#así como el número total de neuronas (na_duracion):
dim_entrada_duracion = (X_train_duracion.shape[1],2)
dim_salida_duracion = 2
na_duracion = 100

#Cear un contenedor usando el módulo Sequential:
modelo_duracion = Sequential()
#añadimos el modelo
modelo_duracion.add(LSTM(units=na_duracion, return_sequences=True, input_shape=dim_entrada_duracion))
#agregamos una capa lSTM
modelo_duracion.add(LSTM(units=na_duracion))
#evitamos el sobreentrtenamiento con dropout
modelo_duracion.add(Dropout(0.2))
#Dense para la capa de salida
modelo_duracion.add(Dense(units=dim_salida_duracion))

#definimos funcion de error y el método para minimizar
modelo_duracion.compile(optimizer='rmsprop', loss='mse')

#implementamos el modelo con 20 iteraciones, epochs
#Presentando a la res lstm lotess de 32 datos
modelo_duracion.fit(X_train_duracion,Y_train_duracion,epochs=100,batch_size=64)

#Guardamos el modelo
modelo_duracion.save('tiempo_entrenado.h5')
import joblib
#guardamos el normalizador
joblib.dump(sc, 'scaler_tiempo.save')