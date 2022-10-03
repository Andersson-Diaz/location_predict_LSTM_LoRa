#librerias necesarias
from tokenize import Number
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
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
#x_test_latitud = set_validacion.iloc[:,4:5]
x_test= np.column_stack((set_validacion.iloc[:,[4]],set_validacion.iloc[:,[5]],set_validacion.iloc[:,[8]],set_validacion.iloc[:,[11]]))

array_latitud = []
for x in range(len(x_test)):
    array_latitud.append(x_test[x,0])

array_longitud = []
for x in range(len(x_test)):
    array_longitud.append(x_test[x,1])

import joblib
scaler = joblib.load('scaler.save')

x_test_n = scaler.transform(x_test)

X_test = []
for i in range(time_step,len(x_test_n)):
    X_test.append(x_test_n[i-time_step:i,0:5])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],4))

new_model = keras.models.load_model('path_to_my_model.h5')
prediccion = new_model.predict(X_test)
prediccion = scaler.inverse_transform(prediccion)


array_latitud_p = []
for x in range(len(prediccion)):
    array_latitud_p.append(prediccion[x,0])

array_longitud_p = []
for x in range(len(prediccion)):
    array_longitud_p.append(prediccion[x,1])

# Funciones auxiliares
def graficar_predicciones(real, prediccion,valor):
    plt.plot(real[time_step:len(array_latitud)], color='red', label='Valor real')
    plt.plot(prediccion, color='blue', label='Predicción')
   # plt.ylim(-44,-43)
    plt.xlabel('Posición')
    plt.ylabel('Grados')
    plt.title(valor)
    plt.legend()
    plt.show()

# Graficar resultados
graficar_predicciones(array_latitud,array_latitud_p,'Test latitud')

graficar_predicciones(array_longitud,array_longitud_p,'Test longitud')
def haversine(lat1, lon1, lat2, lon2):
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia=2*R*math.asin(math.sqrt(a))
    return distancia
from sklearn.metrics import mean_squared_error
mse_lat = mean_squared_error(array_latitud[time_step:],array_latitud_p, squared=False)
mse_lon = mean_squared_error(array_longitud[time_step:],array_longitud_p, squared=False)
print(haversine(0.00000,0.00000,mse_lat,mse_lon))
#Prediccion con los primeros datos del set de validacion
# Using predicted values to predict next step
X_pred = x_test_n.copy()
for i in range(time_step,len(X_pred)):
    xin = (np.column_stack((X_pred[i-time_step:i,0:2],x_test_n[i-time_step:i,2:3],x_test_n[i-time_step:i,3:4]))).reshape(1, time_step, 4)
    X_pred[i] = new_model.predict(xin)


prediccion2 = scaler.inverse_transform(X_pred)

# Funciones auxiliares
def graficar_predicciones2(real, prediccion,campo):
    plt.plot(real[0:len(real)], color='red', label='Valor real')
    plt.plot(prediccion, color='blue', label='Predicción')
    #plt.ylim(-44,-43)
    plt.xlabel('Posición')
    plt.ylabel('Grados')
    plt.title(campo)
    plt.legend()
    plt.show()


# Graficar resultados de prediccion de valores futuros
graficar_predicciones2(array_latitud,prediccion2[:,0:1],'Predicción latitud')

graficar_predicciones2(array_longitud,prediccion2[:,1:2],'Predicción longitud')
mse_lat_p = mean_squared_error(array_latitud[time_step:],prediccion2[time_step:,0:1], squared=False)
mse_lon_p = mean_squared_error(array_longitud[time_step:],prediccion2[time_step:,1:2], squared=False)
print('error medio en prediccion')
print(haversine(0.00000,0.00000,mse_lat_p,mse_lon_p))
#tERMINA PREDICCION DE la ubicacion
#Empieza prediccion de tiempo

#Calcula la diferencia de tiempo entre puntos de ubicacion

time_validacion = set_validacion['hour']

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

duracion_validacion = []
duracion_validacion.append(10)
for i in range(0,len(medida_de_tiempo_validacion)):
    duracion_validacion.append(medida_de_tiempo_validacion[i]/delta)

duracion_validacion
duracion_data_validacion = pd.DataFrame(duracion_validacion, columns=['duracion'])
duracion_data_validacion
#x_test = duracion_data_validacion.values


def haversine(lat1, lon1, lat2, lon2):
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia=2*R*math.asin(math.sqrt(a))
    return distancia

distancia_validacion = []
distancia_validacion.append(0)
for i in range(0, len(set_validacion)-1):
    distancia_validacion.append(haversine(set_validacion.iat[i,4],set_validacion.iat[i,5],set_validacion.iat[i+1,4],set_validacion.iat[i+1,5]))
n_v= np.column_stack((duracion_validacion,distancia_validacion))
import joblib
sc = joblib.load('scaler_tiempo.save')
x_test = sc.transform(n_v)

X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0:2])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],2))

modelo_duracion = keras.models.load_model('tiempo_entrenado.h5')
#Predecimos y normalizamos
prediccion_tiempo = modelo_duracion.predict(X_test)
prediccion_tiempo = sc.inverse_transform(prediccion_tiempo)
