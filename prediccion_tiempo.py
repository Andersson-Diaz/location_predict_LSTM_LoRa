from sqlite3 import TimestampFromTicks
from tokenize import Number
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

import time
import datetime
from datetime import datetime
from datetime import timedelta
import math
import MySQLdb
print('Comienza funcion monitor')
#datos para la conexion a la base de datos
hostname = '82.180.175.58'
username = 'u813407238_lora'
password = 'Seguimiento_lora_123'
database = 'u813407238_seguimiento'  
   

# inicialmente hace la conexion con la base de datos
myConnection = MySQLdb.connect( host=hostname, user=username, passwd=password, db=database )
import pandas as pd
# genera la lectura de la base de datos
dataset= pd.read_sql("SELECT * FROM LoRaWAN_messages order by id ASC LIMIT 152",myConnection)
print("Va a imprimir el dataset leido de la BD...")
print("Va a imprimir date y time...")
print(dataset)

dataset.drop(index=dataset[dataset['latitude']== 0].index, inplace=True)

print(dataset)

set_entrenamiento = dataset[-130:-30]
set_validacion = dataset[-30:]

set_entrenamiento.reset_index(inplace=True, drop=True)
print(set_entrenamiento)

set_validacion.reset_index(inplace=True, drop=True)
print(set_validacion)

#Predicción Latitud:

set_entrenamiento.iloc[:,4:5]
print(set_entrenamiento.iloc[:,4:5])

# Normalización del set de entrenamiento
sc1 = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_latitud_escalado = sc1.fit_transform(set_entrenamiento.iloc[:,4:5])

print(set_entrenamiento_latitud_escalado)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 10
X_train_latitud = []
Y_train_latitud = []
m = len(set_entrenamiento_latitud_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train_latitud.append(set_entrenamiento_latitud_escalado[i-time_step:i,0])
    # Y: el siguiente dato
    Y_train_latitud.append(set_entrenamiento_latitud_escalado[i-time_step,0])

X_train_latitud, Y_train_latitud = np.array(X_train_latitud), np.array(Y_train_latitud)

# Reshape X_train para que se ajuste al modelo en Keras
X_train_latitud = np.reshape(X_train_latitud, (X_train_latitud.shape[0], X_train_latitud.shape[1], 1))

# Red LSTM
#Para crear la red debemos primero definir el tamaño de los datos de entrada y del dato de salida,
#así como el número total de neuronas (50):
dim_entrada = (X_train_latitud.shape[1],1)
dim_salida = 1
na = 100

#Cear un contenedor usando el módulo Sequential:
modelo_latitud = Sequential()
#añadimos el modelo
modelo_latitud.add(LSTM(units=na, input_shape=dim_entrada))
#Dense para la capa de salida
modelo_latitud.add(Dense(units=dim_salida))

#definimos funcion de error y el método para minimizar
modelo_latitud.compile(optimizer='rmsprop', loss='mse')

#implementamos el modelo con 20 iteraciones, epochs
#Presentando a la res lstm lotess de 32 datos
modelo_latitud.fit(X_train_latitud,Y_train_latitud,epochs=100,batch_size=5)
#epochs 100  batch=64 loss 0.0017
#epochs 100  batch=32 loss 0.0040
#epochs 20  batch=32 loss 0.0071

x_test_latitud = set_validacion.iloc[:,4:5]

x_test_latitud = x_test_latitud.values
print(x_test_latitud)

x_test_latitud = sc1.transform(x_test_latitud)
print(x_test_latitud)

#Obtenemos bloques de 60 datos
X_test_latitud = []
for i in range(time_step,len(x_test_latitud)):
    X_test_latitud.append(x_test_latitud[i-time_step:i,0])

X_test_latitud = np.array(X_test_latitud)
print(X_test_latitud.shape)

print(X_test_latitud)

X_test_latitud = np.reshape(X_test_latitud, (X_test_latitud.shape[0],X_test_latitud.shape[1],1))

#Predecimos y normalizamos
prediccion = modelo_latitud.predict(X_test_latitud)
prediccion = sc1.inverse_transform(prediccion)
print(prediccion.shape)

print(prediccion)

# Funciones auxiliares
def graficar_predicciones(real, prediccion):
    plt.plot(real[time_step:len(real)],color='red', label='Valor real')
    plt.plot(prediccion, 'o', color='blue', label='Predicción')
   # plt.ylim(-44,-43)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()

# Graficar resultados
graficar_predicciones(set_validacion.iloc[:,4:5].values,prediccion)

#Prediccion con los primeros datos del set de validacion
# Using predicted values to predict next step
X_pred = x_test_latitud.copy()
for i in range(time_step,len(X_pred)):
    xin = X_pred[i-time_step:i].reshape(1, time_step, 1)
    X_pred[i] = modelo_latitud.predict(xin)

prediccion2 = sc1.inverse_transform(X_pred)

print(prediccion2)

# Funciones auxiliares
def graficar_predicciones2(real, prediccion):
    plt.plot(real, 'o',color='red', label='Valor real')
    plt.plot(prediccion, 'o', color='blue', label='Predicción')
    #plt.ylim(-44,-43)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()

# Graficar resultados de prediccion de valores futuros
graficar_predicciones2(set_validacion.iloc[:,4:5].values,prediccion2)

data_latitud = pd.DataFrame(prediccion2)
print(data_latitud)

##################################################
#Predicción Longitud:
##################################################

set_entrenamiento.iloc[:,5:6]
print(set_entrenamiento.iloc[:,5:6])

# Normalización del set de entrenamiento
sc2 = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_longitud_escalado = sc2.fit_transform(set_entrenamiento.iloc[:,5:6])

print(set_entrenamiento_longitud_escalado)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 10
X_train_longitud = []
Y_train_longitud = []
m = len(set_entrenamiento_longitud_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train_longitud.append(set_entrenamiento_longitud_escalado[i-time_step:i,0])
    # Y: el siguiente dato
    Y_train_longitud.append(set_entrenamiento_longitud_escalado[i-time_step,0])

X_train_longitud, Y_train_longitud = np.array(X_train_longitud), np.array(Y_train_longitud)

# Reshape X_train para que se ajuste al modelo en Keras
X_train_longitud = np.reshape(X_train_longitud, (X_train_longitud.shape[0], X_train_longitud.shape[1], 1))

# Red LSTM
#Para crear la red debemos primero definir el tamaño de los datos de entrada y del dato de salida,
#así como el número total de neuronas (50):
dim_entrada = (X_train_longitud.shape[1],1)
dim_salida = 1
na = 100

#Cear un contenedor usando el módulo Sequential:
modelo_longitud = Sequential()
#añadimos el modelo
modelo_longitud.add(LSTM(units=na, return_sequences=True, input_shape=dim_entrada))

modelo_longitud.add(Dropout(0.2))
modelo_longitud.add(LSTM(units=na))
modelo_longitud.add(Dropout(0.2))
#Dense para la capa de salida
modelo_longitud.add(Dense(units=dim_salida))

#definimos funcion de error y el método para minimizar
modelo_longitud.compile(optimizer='rmsprop', loss='mse')

#implementamos el modelo con 20 iteraciones, epochs
#Presentando a la res lstm lotess de 32 datos
modelo_longitud.fit(X_train_longitud,Y_train_longitud,epochs=100,batch_size=5)
#epochs 100  batch=64 loss 0.0017
#epochs 100  batch=32 loss 0.0040
#epochs 20  batch=32 loss 0.0071

x_test_longitud = set_validacion.iloc[:,5:6]

x_test_longitud = x_test_longitud.values
print(x_test_longitud)

x_test_longitud = sc2.transform(x_test_longitud)
print(x_test_longitud)

#Obtenemos bloques de 60 datos
X_test_longitud = []
for i in range(time_step,len(x_test_longitud)):
    X_test_longitud.append(x_test_longitud[i-time_step:i,0])

X_test_longitud = np.array(X_test_longitud)
print(X_test_longitud.shape)

print(X_test_longitud)

X_test_longitud = np.reshape(X_test_longitud, (X_test_longitud.shape[0],X_test_longitud.shape[1],1))

#Predecimos y normalizamos
prediccion = modelo_longitud.predict(X_test_longitud)
prediccion = sc2.inverse_transform(prediccion)
print(prediccion.shape)

print(prediccion)

# Funciones auxiliares
def graficar_predicciones(real, prediccion):
    plt.plot(real[0:len(real)],color='red', label='Valor real')
    plt.plot(prediccion, color='blue', label='Predicción')
    #plt.ylim(-77,-75)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()

# Graficar resultados
graficar_predicciones(set_validacion.iloc[:,5:6].values,prediccion)

#Prediccion con los primeros datos del set de validacion
# Using predicted values to predict next step
X_pred = x_test_longitud.copy()
for i in range(time_step,len(X_pred)):
    xin = X_pred[i-time_step:i].reshape(1, time_step, 1)
    X_pred[i] = modelo_longitud.predict(xin)

prediccion2 = sc2.inverse_transform(X_pred)

print(prediccion2)

# Funciones auxiliares
def graficar_predicciones2(real, prediccion):
    plt.plot(real,'o', color='red', label='Valor real')
    plt.plot(prediccion, 'o', color='blue', label='Predicción')
    #plt.ylim(-77,-75)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()

# Graficar resultados de prediccion de valores futuros
graficar_predicciones2(set_validacion.iloc[:,5:6].values,prediccion2)

data_longitud = pd.DataFrame(prediccion2)
print(data_longitud)


########################################################
#Predecir tiempo 
########################################################

time_entrenamiento = set_entrenamiento['hour']
print("set entrenamiento:")
print(time_entrenamiento.head(10))

time_validacion = set_validacion['hour']
print("set validacion:")
print(time_validacion)

#Calcula la diferencia de tiempo entre puntos de ubicacion
medida_de_tiempo_entrenamiento = []
for i in range(0,len(time_entrenamiento)-1):
    medida_de_tiempo_entrenamiento.append(time_entrenamiento[i+1]-time_entrenamiento[i])
print(medida_de_tiempo_entrenamiento)

#Calcula la diferencia de tiempo entre puntos de ubicacion
medida_de_tiempo_validacion = []
for i in range(0,len(time_validacion)-1):
    medida_de_tiempo_validacion.append(time_validacion[i+1]-time_validacion[i])
print(medida_de_tiempo_validacion)

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

duracion_entrenamiento = []
#duracion_entrenamiento.append(0)
for i in range(0,len(medida_de_tiempo_entrenamiento)):
    duracion_entrenamiento.append(medida_de_tiempo_entrenamiento[i]/delta)
print(duracion_entrenamiento)

duracion_validacion = []
#duracion_validacion.append(0)
for i in range(0,len(medida_de_tiempo_validacion)):
    duracion_validacion.append(medida_de_tiempo_validacion[i]/delta)
print(duracion_validacion)

duracion_data_entrenamiento = pd.DataFrame(duracion_entrenamiento, columns=['duracion'])
print(duracion_data_entrenamiento)

duracion_data_validacion = pd.DataFrame(duracion_validacion, columns=['duracion'])
print(duracion_data_validacion)

#normalizamos
sc = MinMaxScaler(feature_range=(0,1))
time_entrenamiento_escalado = sc.fit_transform(duracion_data_entrenamiento)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 10
X_train_duracion = []
Y_train_duracion = []
n = len(duracion_data_entrenamiento)

for i in range(time_step,n):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train_duracion.append(time_entrenamiento_escalado[i-time_step:i,0])
    # Y: el siguiente dato
    Y_train_duracion.append(time_entrenamiento_escalado[i,0])

X_train_duracion, Y_train_duracion = np.array(X_train_duracion), np.array(Y_train_duracion)

    
# Reshape X_train para que se ajuste al modelo en Keras
X_train_duracion = np.reshape(X_train_duracion, (X_train_duracion.shape[0], X_train_duracion.shape[1], 1))
# Red LSTM
#Para crear la red debemos primero definir el tamaño de los datos de entrada y del dato de salida,
#así como el número total de neuronas (50):
dim_entrada_duracion = (X_train_duracion.shape[1],1)
dim_salida_duracion = 1
na_duracion = 100

#Cear un contenedor usando el módulo Sequential:
modelo_duracion = Sequential()
#añadimos el modelo
modelo_duracion.add(LSTM(units=na_duracion, input_shape=dim_entrada_duracion))
#Dense para la capa de salida
modelo_duracion.add(Dense(units=dim_salida_duracion))

#definimos funcion de error y el método para minimizar
modelo_duracion.compile(optimizer='rmsprop', loss='mse')

#implementamos el modelo con 20 iteraciones, epochs
#Presentando a la res lstm lotess de 32 datos
modelo_duracion.fit(X_train_duracion,Y_train_duracion,epochs=100,batch_size=64)

x_test = duracion_data_validacion.values
print(x_test)

x_test = sc.transform(x_test)
print(x_test)

#Obtenemos bloques de 60 datos
X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])

X_test = np.array(X_test)
print(X_test.shape)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

#Predecimos y normalizamos
prediccion_t = modelo_duracion.predict(X_test)
prediccion_t = sc.inverse_transform(prediccion_t)


# Funciones auxiliares
def graficar_predicciones(real, prediccion):
    plt.plot(real[0:len(prediccion)],color='red', label='Valor real')
    plt.plot(prediccion, color='blue', label='Predicción')
   # plt.ylim(-44,-43)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()

# Graficar resultados
graficar_predicciones(duracion_data_validacion.values,prediccion_t)

#Prediccion con los primeros datos del set de validacion
# Using predicted values to predict next step
X_pred = x_test.copy()
for i in range(time_step,len(X_pred)):
    xin = X_pred[i-time_step:i].reshape(1, time_step, 1)
    X_pred[i] = modelo_duracion.predict(xin)

prediccion_t2 = sc.inverse_transform(X_pred)

print(prediccion_t2)

# Funciones auxiliares
def graficar_predicciones2(real, prediccion):
    plt.plot(real,'o', color='red', label='Valor real')
    plt.plot(prediccion, 'o', color='blue', label='Predicción')
    #plt.ylim(-77,-75)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()

# Graficar resultados de prediccion de valores futuros
graficar_predicciones2(duracion_data_validacion.values,prediccion_t2)

data_tiempo = pd.DataFrame(prediccion_t2)
print(data_tiempo)

#print(duracion_data_validacion)

latitud_longitud_tiempo = pd.concat([data_latitud, data_longitud, data_tiempo], axis=1)
print(latitud_longitud_tiempo)
print()

#Agrego los segundos a una hora de referencia
hora = time_validacion[9]
suma_tiempo = time_validacion.copy()
for i in range(time_step,len(suma_tiempo)):
    suma_tiempo[i] = hora + timedelta(seconds=latitud_longitud_tiempo.iat[i-1,2])
    print("suma tiempo:")
    print(suma_tiempo[i])
    hora = suma_tiempo[i]
    print("hora:")
    print(hora)

print()    
print(suma_tiempo)

import datetime
from datetime import datetime
from datetime import timedelta
hora_sistema = datetime.now()
print(hora_sistema)

hora_sistema = timedelta(hours=hora_sistema.hour, minutes=hora_sistema.minute, seconds=hora_sistema.second, microseconds=hora_sistema.microsecond)
print(hora_sistema)

diferencia_tiempo = []
for i in range(0,len(suma_tiempo)):
    diferencia_tiempo.append(hora_sistema - suma_tiempo[i])

print(diferencia_tiempo)

index = diferencia_tiempo.index(min(diferencia_tiempo))
print(index)
print(latitud_longitud_tiempo.iat[index,0], latitud_longitud_tiempo.iat[index,1])

'''print(len(latitud_longitud))
for recorrido in range(len(latitud_longitud)):
    print("latitud:",latitud_longitud.iat[recorrido,0],"longitud:",latitud_longitud.iat[recorrido,1])'''


mynewConnection = MySQLdb.connect( host=hostname, user=username, passwd=password, db=database )
cur = mynewConnection.cursor()
from datetime import datetime
now = datetime.now()
fecha = format(now.date()) 
hora = format(now.time())
latitud_bd = latitud_longitud_tiempo.iat[index,0]
longitud_bd = latitud_longitud_tiempo.iat[index,1]
cadena_SQL= "INSERT INTO predicciones (latitude, longitude) VALUES (%s,%s)"
val=(latitud_bd,longitud_bd)
#print (cadena_SQL)
cur.execute(cadena_SQL,val)
print("Registro creado.")
    
# Si no se graba, no guarda el cambio de la creacion, guarda con commit
mynewConnection.commit()    
# Cierra la conexion
mynewConnection.close()