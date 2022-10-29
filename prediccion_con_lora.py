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
dataset['latitude']=dataset['latitude'].astype('float64')
dataset['longitude']=dataset['longitude'].astype('float64')
window_time = 30
#Obtenemos
last = int(len(dataset)/5.0)
set_validation = dataset[-last-window_time:]
set_validation.reset_index(inplace=True, drop=True)
#x_test_latitud = set_validation.iloc[:,4:5]
x_test= np.column_stack((set_validation.iloc[:,[4]],set_validation.iloc[:,[5]],set_validation.iloc[:,[8]],set_validation.iloc[:,[11]]))

array_latitud = []
for x in range(len(x_test)):
    array_latitud.append(x_test[x,0])

array_longitud = []
for x in range(len(x_test)):
    array_longitud.append(x_test[x,1])

import joblib
scaler = joblib.load('scaler.save')

x_test_normalized = scaler.transform(x_test)

X_test = []
for i in range(window_time,len(x_test_normalized)):
    X_test.append(x_test_normalized[i-window_time:i,0:5])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],4))

new_model = keras.models.load_model('model_LSTM.h5')
#new_model = keras.models.load_model('model_bidirectional.h5')
prediction = new_model.predict(X_test)
prediction = scaler.inverse_transform(prediction)


array_latitud_predict = []
for x in range(len(prediction)):
    array_latitud_predict.append(prediction[x,0])

array_longitud_predict = []
for x in range(len(prediction)):
    array_longitud_predict.append(prediction[x,1])

# Funciones auxiliares
def plot_prediction(real, prediction,value,y):
    plt.plot(real[window_time:len(array_latitud)], color='red', label='Valor real')
    plt.plot(prediction, color='blue', label='Predicción')
   # plt.ylim(-44,-43)
    plt.xlabel('Número de mediciones')
    plt.ylabel(y)
    plt.title(value)
    plt.legend()
    plt.show()

# Graficar resultados
plot_prediction(array_latitud,array_latitud_predict,'Test de latitud','Latitud(Grados)')
plot_prediction(array_longitud,array_longitud_predict,'Test de longitud','Longitud(Grados)')

def haversine(lat1, lon1, lat2, lon2):
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distance=2*R*math.asin(math.sqrt(a))
    return distance

from sklearn.metrics import mean_squared_error
mse_lat = mean_squared_error(array_latitud[window_time:],array_latitud_predict, squared=False)
mse_lon = mean_squared_error(array_longitud[window_time:],array_longitud_predict, squared=False)
print('mse_lat',mse_lat)
print('mse_lom',mse_lon)
print('mse: error medio en predicción, en valores históricos')
print(haversine(0.00000,0.00000,mse_lat,mse_lon),' kilometros')
#Prediction con los primeros datos del set de validacion
# Using predicted values to predict next step
X_pred = x_test_normalized.copy()
for i in range(window_time,len(X_pred)):
    xin = (np.column_stack((X_pred[i-window_time:i,0:2],x_test_normalized[i-window_time:i,2:3],x_test_normalized[i-window_time:i,3:4]))).reshape(1, window_time, 4)
    X_pred[i] = new_model.predict(xin)

prediction_test = scaler.inverse_transform(X_pred)

# Funciones auxiliares
def plot_time_prediction(real, prediction,title,y):
    plt.plot(real[0:len(real)], color='red', label='Valor real')
    plt.plot(prediction, color='blue', label='Predicción')
    #plt.ylim(-44,-43)
    plt.xlabel('Número de mediciones')
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.show()

# Graficar resultados de prediction de valores futuros
plot_time_prediction(array_latitud,prediction_test[:,0:1],'Predicción de latitud','Latitud(Grados)')
plot_time_prediction(array_longitud,prediction_test[:,1:2],'Predicción de longitud','Longitud(Grados)')

mse_lat_p = mean_squared_error(array_latitud[window_time:],prediction_test[window_time:,0:1], squared=False)
mse_lon_p = mean_squared_error(array_longitud[window_time:],prediction_test[window_time:,1:2], squared=False)
print('mse_lat',mse_lat_p)
print('mse_lom',mse_lon_p)
print('error medio en prediccion')
print(haversine(0.00000,0.00000,mse_lat_p,mse_lon_p),' kilometros')

calle_5_p = pd.DataFrame(prediction_test[:,0:2])
calle_5_p.to_csv('calle_5_con_lora_prediccion.csv')
#tERMINA PREDICtion DE la ubicacion
#Empieza prediction de tiempo

#Calcula la diferencia de tiempo entre puntos de ubicacion

time_validation = set_validation['hour']

time_validation_measured = []
for i in range(0,len(time_validation)-1):
    time_validation_measured.append(time_validation[i+1]-time_validation[i])

#Creamos un objeto deltatime de value 1 segundo
#Al dividir deltatime / deltatime se obtiene un value de tipo float
#Al dividir sobre un segundo se obtiene un value de tiempo en segundos
delta = timedelta(
    days=0,
    seconds=1,
    microseconds=0,
    milliseconds=0,
    minutes=0,
    hours=0,
    weeks=0 )

validation_last = []
validation_last.append(10)
for i in range(0,len(time_validation_measured)):
    validation_last.append(time_validation_measured[i]/delta)

validation_last
last_data_validation = pd.DataFrame(validation_last, columns=['duracion'])
last_data_validation
#x_test = last_data_validation.values

def haversine(lat1, lon1, lat2, lon2):
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distance=2*R*math.asin(math.sqrt(a))
    return distance

distance_validation = []
distance_validation.append(0)
for i in range(0, len(set_validation)-1):
    distance_validation.append(haversine(set_validation.iat[i,4],set_validation.iat[i,5],set_validation.iat[i+1,4],set_validation.iat[i+1,5]))
n_v= np.column_stack((validation_last,distance_validation))
import joblib
sc = joblib.load('scaler_tiempo.save')
x_test = sc.transform(n_v)

window_time = 30
X_test = []
for i in range(window_time,len(x_test)):
    X_test.append(x_test[i-window_time:i,0:2])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],2))

model_last = keras.models.load_model('tiempo_entrenado.h5')
#Predecimos y normalizamos
prediction_test = model_last.predict(X_test)
prediction_test = sc.inverse_transform(prediction_test)
# Funciones auxiliares
def graficar_predicciones3(real, prediccion,campo):
    plt.plot(real[0:len(prediccion)], 'o',   color='red', label='Valor real')
    plt.plot(prediccion,'o', color='blue', label='Predicción')
    #plt.ylim(9,15)
    #plt.xlim(40,50)
    plt.xlabel('Número de medición')
    plt.ylabel(campo)
    plt.title('Predicción de tiempo entre mediciones')
    plt.legend()
    plt.show()

graficar_predicciones3(n_v[:,0:1],prediction_test[:,0:1],'Tiempo(segundos)')
mse_t = mean_squared_error(n_v[window_time:,0:1] ,prediction_test[:,0:1], squared=False)
print('Error medio en tiempo predicho: ',mse_t)

n_v_t = sc.transform(n_v)
X_pred_t = n_v_t.copy()
for i in range(window_time,len(X_pred_t)):
    xin_t = X_pred_t[i-window_time:i].reshape(1, window_time, 2)
    X_pred_t[i] = model_last.predict(xin_t)

prediction_time = sc.inverse_transform(X_pred_t)

# Graficar resultados de prediccion de valores futuros
graficar_predicciones3(n_v[:,0:1],prediction_time[:,0:1],'Tiempo(segundos)')
mse_t_p = mean_squared_error(n_v[window_time:,0:1] ,prediction_time[window_time:,0:1], squared=False)
print('Error medio en tiempo predicho: ',mse_t_p)