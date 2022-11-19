#from tokenize import Number
import numpy as np
#np.random.seed(83)
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.random.set_seed(100)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
#import tensorflow as tf
#print('Version de tensorflow ',tf.__version__)
#from keras.utils import to_categorical
from keras.layers import Flatten
import time
from datetime import datetime
from datetime import timedelta
import math
from keras.optimizers import SGD, Adam
from keras.layers import Flatten

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

dataset.drop(index=dataset[dataset['latitude']=='0'].index, inplace=True)
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
# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)
for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0:5])
    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0:5])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
# Red LSTM
#Para crear la red debemos primero definir el tamaño de los datos de entrada y del dato de salida,
#así como el número total de neuronas (100):
dim_in = (X_train.shape[1],4)
dim_out = 4
neurons = 100
modelo = Sequential()
modelo.add(LSTM(units=200, activation = 'relu',  return_sequences=True, input_shape=dim_in))
modelo.add(LSTM(200, activation = 'relu',return_sequences=True))
modelo.add(LSTM(50, activation = 'relu',return_sequences=True))
#modelo.add(LSTM(200, input_shape=dim_in))
modelo.add(Flatten())
modelo.add(Dropout(0.0))
opt = Adam(learning_rate=0.001)
modelo.add(Dense(units=dim_out ,activation = 'relu'))
modelo.compile(optimizer=opt, loss='mse', metrics = 'mse')
historia = modelo.fit(X_train,Y_train,epochs=500,batch_size=32)
print(modelo.summary())
x_test= np.column_stack((set_validacion.iloc[:,[4]],set_validacion.iloc[:,[5]],set_validacion.iloc[:,[8]],set_validacion.iloc[:,[12]]))
array_latitud = []
for x in range(len(x_test)):
    array_latitud.append(x_test[x,0])

array_longitud = []
for x in range(len(x_test)):
    array_longitud.append(x_test[x,1])

x_test_n = sc1.transform(x_test)

#Obtenemos bloques de 60 datos
X_test = []
for i in range(time_step,len(x_test_n)):
    X_test.append(x_test_n[i-time_step:i,0:5])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],4))
prediccion = modelo.predict(X_test)
prediccion = sc1.inverse_transform(prediccion)
array_latitud_p = []
for x in range(len(prediccion)):
    array_latitud_p.append(prediccion[x,0])

array_longitud_p = []
for x in range(len(prediccion)):
    array_longitud_p.append(prediccion[x,1])

# Funciones auxiliares
def graficar_predicciones(real, prediccion,titulo):
    plt.plot(real[time_step:len(array_latitud)], color='red', label='Valor real')
    plt.plot(prediccion, color='blue', label='Predicción')
   # plt.ylim(-44,-43)
    plt.xlabel('Número de mediciones')
    plt.ylabel('Grados')
    plt.title(titulo)
    plt.legend()
    plt.show()

from sklearn.metrics import mean_squared_error
mse_lat = mean_squared_error(array_latitud[time_step:],array_latitud_p, squared=False)
print('mse lat: ', mse_lat)
mse_lon = mean_squared_error(array_longitud[time_step:],array_longitud_p, squared=False)
print('mse lon: ', mse_lon)
def haversine(lat1, lon1, lat2, lon2):
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia=2*R*math.asin(math.sqrt(a))
    return distancia
print('error medio: ',haversine(0.00000,0.00000,mse_lat,mse_lon))

#Prediccion con los primeros datos del set de validacion
# Using predicted values to predict next step
X_pred = x_test_n.copy()
for i in range(time_step,len(X_pred)):
    xin = X_pred[i-time_step:i].reshape(1, time_step, 4)
    X_pred[i] = modelo.predict(xin)

prediccion2 = sc1.inverse_transform(X_pred)
# Funciones auxiliares
def graficar_predicciones2(real, prediccion,campo,title):
    plt.plot(real[0:len(real)], color='red', label='Valor real')
    plt.plot(prediccion, color='blue', label='Predicción')
    #plt.ylim(-44,-43)
    plt.title(title)
    plt.xlabel('Numero de mediciones')
    plt.ylabel(campo)
    plt.legend()
    plt.show()

mse_lat_p = mean_squared_error(array_latitud[time_step:],prediccion2[time_step:,0:1], squared=False)
print('mse lat_p : ', mse_lat_p)

mse_lon_p = mean_squared_error(array_longitud[time_step:],prediccion2[time_step:,1:2], squared=False)
print('mse lon_p : ', mse_lon_p)

print('error medio predicho: ',haversine(0.00000,0.00000,mse_lat_p,mse_lon_p))

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(historia.history['mse'])
plt.title('mse')
plt.xlabel('epochs')
plt.subplot(1,2,2)
plt.plot(historia.history['loss'],color='orange')
plt.title('loss')
plt.xlabel('epochs')
plt.show()