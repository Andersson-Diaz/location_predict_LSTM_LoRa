
#librerias necesarias
#from tokenize import Number
import numpy as np
from numpy import *
np.random.seed(4)

import tensorflow as tf
#tf.random.set_seed(100)
#import matplotlib.pyplot as plt
import keras_tuner
import pandas as pd
from tensorflow import keras
from keras.layers import Flatten
#from keras.optimizers import SGD, Adam


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
#from keras.layers import Bidirectional
import time
inicio = time.time()
#from datetime import datetime
from datetime import timedelta
import math
import MySQLdb
import pandas as pd

#datos para la conexion a la base de datos
hostname = '82.180.175.58'
username = 'u813407238_lora'
password = 'Seguimiento_lora_123'
database = 'u813407238_seguimiento'
#Longitud de entrenamiento
len_training = 100

def ejecutar_entrenamiento(r,t,data):
    window = 30
    dataset = data
    last = int(len(dataset)/5.0)
    import pandas as pd
    #iron = pd.read_csv('valor_final_inicio_entrenamiento.csv')
    #ir = iron.iloc[0,1]
    
    #set_total = dataset[r:(r +len(dataset)-r)]
    set_total = dataset[r:(len(dataset))]

    set_total.reset_index(inplace=True, drop=True)
    set_training = set_total[:-last]
    set_validation = set_total[-last-window:]
    set_training.reset_index(inplace=True, drop=True)
    set_validation.reset_index(inplace=True, drop=True)

    x= np.column_stack((set_training.iloc[:,[4]],set_training.iloc[:,[5]],set_training.iloc[:,[8]],set_training.iloc[:,[12]]))
    # Normalización del set de entrenamiento
    scaler = MinMaxScaler(feature_range=(0,1))
    set_training_escaled = scaler.fit_transform(x)
    #Declaracion de vectores de entrada y salida para el entrenamiento
    X_train = []
    Y_train = []
    m = len(set_training_escaled)
    #Crea las ventanas de datos
    for i in range(window,m):
        # X: bloques de "window" datos: 0-window, 1-window+1, 2-window+2, etc
        X_train.append(set_training_escaled[i-window:i,0:5])
        # Y: el siguiente dato despues de la ventana de datos
        Y_train.append(set_training_escaled[i,0:5])
    #Transforma las listas en vectores
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    #Reforma el vector para que se ajuste al modelo en keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
    # Red LSTM
    #Para crear la red debemos primero definir el tamaño de los datos de entrada y del dato de salida,
    #así como el número total de neurons (100):
    dim_in = (X_train.shape[1],4)
    dim_out = 4
    neurons = 200  

    """modelo = Sequential()
    modelo.add(LSTM(units=neurons,  return_sequences=True, input_shape=dim_in))
    modelo.add(LSTM(200,  return_sequences=True))
    modelo.add(LSTM(100,  return_sequences=True))
    modelo.add(LSTM(100,  return_sequences=True))
    modelo.add(Flatten())
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=dim_out))
    modelo.compile(optimizer='Adam', loss='mse')
    modelo.fit(X_train,Y_train,epochs=100,batch_size=32)"""
    from keras.layers import Bidirectional
    
    from tensorflow.keras.optimizers import SGD, Adam

    def build_model(hp):
    #definicion de hiperparámetros a evaluar
        hp_batch_size = hp.Int('batch_size', min_value = 8, max_value = 128, step = 8)
        hp_seed = hp.Int('seed', min_value =0, max_value = 100, step = 1)
        hp_activation = hp.Choice('activation',['relu','tanh','linear','selu','elu','softmax'])
        recurrent_dropout = hp.Float('recurrent_dropout',min_value=0.0,max_value=0.99,default=0.2)
        hp_neurons = hp.Int('neurons1', min_value = 10, max_value = 200, step = 10)
        hp_neurons2 = hp.Int('neurons2', min_value = 10, max_value = 200, step = 10)
        hp_neurons3 = hp.Int('neurons3', min_value = 10, max_value = 200, step = 10)
        tf.random.set_seed(hp_seed)
        #definición del modelo de red neuronal
        modelo = Sequential()
        modelo.add(Bidirectional(LSTM(units=hp_neurons, activation=hp_activation, return_sequences=True, input_shape=dim_in)))
        modelo.add(Bidirectional(LSTM(hp_neurons2, activation=hp_activation, return_sequences=True)))
        modelo.add(Bidirectional(LSTM(hp_neurons3, activation=hp_activation, return_sequences=True)))
        #modelo.add(LSTM(200, activation=hp_activation,  return_sequences=True))
        modelo.add(Flatten())
        modelo.add(Dropout(recurrent_dropout))
        modelo.add(Dense(units=dim_out,activation=hp_activation))
    
        hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3])
        opt = Adam(learning_rate=hp_learning_rate)#, momentum=0.9)
        modelo.compile(optimizer=opt, loss='mse', metrics=['mse'])
        #modelo.compile(optimizer='Adam', loss='mse')
        #modelo.fit(X_train,Y_train,epochs=10,batch_size=32)
        #print(modelo.summary())
        return modelo
    
    #from keras_tuner.tuners import BayesianOptimization

    tuner = keras_tuner.BayesianOptimization(
    build_model,
    objective = 'mse',
    overwrite=True,
    max_trials = 6)

    x_test= np.column_stack((set_validation.iloc[:,[4]],set_validation.iloc[:,[5]],set_validation.iloc[:,[8]],set_validation.iloc[:,[12]]))

    x_test_n = scaler.transform(x_test)

    X_test = []
    Y_test = []
    for i in range(window,len(x_test_n)):
        X_test.append(x_test_n[i-window:i,0:5])
        Y_test.append(x_test_n[i,0:5])

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],4))

    tuner.search(X_train, Y_train,
             epochs=100,
             #overwrite=True,
             validation_data=(X_test, Y_test))

    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    print(best_hps.values)

    modelo = tuner.hypermodel.build(best_hps)
    modelo.fit(X_train, Y_train,
                      epochs=100,batch_size=32,
                      validation_data=(X_test, Y_test))
    
    #print(modelo.summary())
    modelo.save('model_LSTM_tarjeta2.h5')
    import joblib
    joblib.dump(scaler, 'scaler_tarjeta2.save')                
    print('Entrenamiento ejecutado, valor de inicio: ', r, 'valor final: ', t)
    
    #fin ejecutar entrenamiento para la posicion
    ###########################
    #inicio ejecutar entrenamiento para el tiempo
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
    training_distance = []   
    for i in range(0, len(set_training)-1):
        training_distance.append(haversine(set_training.iat[i,4],set_training.iat[i,5],set_training.iat[i+1,4],set_training.iat[i+1,5]))
    
    validation_distance = []
    for i in range(0, len(set_validation)-1):
        validation_distance.append(haversine(set_validation.iat[i,4],set_validation.iat[i,5],set_validation.iat[i+1,4],set_validation.iat[i+1,5]))
    
    #toma la hora del conjunto de datos de entrenamiento
    training_time = set_training['hour']
    validation_time = set_validation['hour']
    #Calcula la diferencia de tiempo entre un punto de ubicacion y el anterior en el set de entrenamiento
    training_time_measure = []
    for i in range(0,len(training_time)-1):
        training_time_measure.append(training_time[i+1]-training_time[i])    

    validation_time_measure = []
    for i in range(0,len(validation_time)-1):
        validation_time_measure.append(validation_time[i+1]-validation_time[i])
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
    set_training_in_seconds = []    
    for i in range(0,len(training_time_measure)):
        set_training_in_seconds.append(training_time_measure[i]/delta)    
    #Para la validacion
    set_validation_in_seconds = []    
    for i in range(0,len(validation_time_measure)):
        set_validation_in_seconds.append(validation_time_measure[i]/delta) 
    #Agrupamos tiempo y distancia para usarlas como entradas del modelo de entrenamiento y validacion
    training_stack_in= np.column_stack((set_training_in_seconds,training_distance))
    validation_stack_in= np.column_stack((set_validation_in_seconds,validation_distance))

    #normalizamos el conjunto de datos de entrada
    sc = MinMaxScaler(feature_range=(0,1))
    set_training_time_scaled = sc.fit_transform(training_stack_in)
    set_validation_time_scaled = sc.fit_transform(validation_stack_in)


    # La red LSTM tendrá como entrada "window" datos consecutivos, y como salida 1 dato (la predicción se hace a
    # partir de esos "window" datos). Se conformará de esta forma el set de entrenamiento y validacion
    window_t = 30
    X_train_time = []
    Y_train_time = []
    X_test_time = []
    Y_test_time = []

    #Se agregan las ventanas del set de entrenamiento
    for i in range(window_t,len(training_stack_in)):
        # X: bloques de "window" datos: 0-window, 1-window+1, 2-window+2, etc
        X_train_time.append(set_training_time_scaled[i-window_t:i,0:2])
        # Y: el siguiente dato
        Y_train_time.append(set_training_time_scaled[i,0:2])

    #Se agregan las ventanas del set de validacion
    for i in range(window_t,len(validation_stack_in)):
        # X: bloques de "window" datos: 0-window, 1-window+1, 2-window+2, etc
        X_test_time.append(set_validation_time_scaled[i-window_t:i,0:2])
        # Y: el siguiente dato
        Y_test_time.append(set_validation_time_scaled[i,0:2])

    #Se pasan las listas a arrays
    X_train_time, Y_train_time = np.array(X_train_time), np.array(Y_train_time)
    X_test_time, Y_test_time = np.array(X_test_time), np.array(Y_test_time)

    # Reshape X_train para que se ajuste al modelo en Keras
    X_train_time = np.reshape(X_train_time, (X_train_time.shape[0], X_train_time.shape[1], 2))
    X_test_time = np.reshape(X_test_time, (X_test_time.shape[0], X_test_time.shape[1], 2))

    # Red LSTM
    #Para crear la red debemos primero definir el tamaño de los datos de entrada y del dato de salida,
    #así como el número total de neurons (neurons_time):
    dim_time_in = (X_train_time.shape[1],2)
    dim_time_out = 2
    neurons_time = 100
    #Cear un contenedor usando el módulo Sequential:
    def build_model_time(hp):
    #definicion de hiperparámetros a evaluar
        hp_batch_size = hp.Int('batch_size', min_value = 8, max_value = 128, step = 8)
        #hp_seed = hp.Int('seed', min_value =0, max_value = 100, step = 1)
        hp_activation = hp.Choice('activation',['relu','tanh','linear','selu','elu','softmax'])
        recurrent_dropout = hp.Float('recurrent_dropout',min_value=0.0,max_value=0.99,default=0.2)
        hp_neurons = hp.Int('neurons1', min_value = 10, max_value = 200, step = 10)
        hp_neurons2 = hp.Int('neurons2', min_value = 10, max_value = 200, step = 10)
        hp_neurons3 = hp.Int('neurons3', min_value = 10, max_value = 200, step = 10)
        #tf.random.set_seed(hp_seed)
        #definición del modelo de red neuronal
        model_time = Sequential()
        model_time.add(LSTM(units=hp_neurons, activation=hp_activation, return_sequences=True, input_shape=dim_time_in))
        model_time.add(LSTM(hp_neurons2, activation=hp_activation, return_sequences=True))
        model_time.add(LSTM(hp_neurons3, activation=hp_activation, return_sequences=True))
        #model_time.add(LSTM(200, activation=hp_activation,  return_sequences=True))
        model_time.add(Flatten())
        model_time.add(Dropout(recurrent_dropout))
        model_time.add(Dense(units=dim_time_out,activation=hp_activation))
    
        hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3])
        opt = Adam(learning_rate=hp_learning_rate)#, momentum=0.9)
        model_time.compile(optimizer=opt, loss='mse', metrics=['mse'])
        #model_time.compile(optimizer='Adam', loss='mse')
        #model_time.fit(X_train,Y_train,epochs=10,batch_size=32)
        #print(model_time.summary())
        return model_time

    tuner = keras_tuner.BayesianOptimization(
    build_model_time,
    objective = 'mse',
    overwrite=True,
    max_trials = 6)

    tuner.search(X_train_time, Y_train_time,
             epochs=100,
             #overwrite=True,
             validation_data=(X_test_time, Y_test_time))
    
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    print(best_hps.values)

    model_time = tuner.hypermodel.build(best_hps)
    model_time.fit(X_train_time, Y_train_time,
                      epochs=200,batch_size=32,
                      validation_data=(X_test_time, Y_test_time))

    """model_time = Sequential()
    #añadimos el modelo
    model_time.add(LSTM(units=neurons_time, return_sequences=True, input_shape=dim_time_in))
    #agregamos una capa lSTM
    model_time.add(LSTM(units=neurons_time))
    #evitamos el sobreentrtenamiento con dropout
    model_time.add(Dropout(0.2))
    #Dense para la capa de salida
    model_time.add(Dense(units=dim_time_out))
    #definimos funcion de error y el método para minimizar
    model_time.compile(optimizer='rmsprop', loss='mse')
    #implementamos el modelo con 20 iteraciones, epochs
    #Presentando a la res lstm lotes de 32 datos
    model_time.fit(X_train_time,Y_train_time,epochs=100,batch_size=32)"""
    #Guardamos el modelo
    model_time.save('tiempo_entrenado_tarjeta2.h5')
    import joblib
    #guardamos el normalizador
    joblib.dump(sc, 'scaler_tiempo_tarjeta2.save')   
    fin = time.time()
    print('tiempo de ejecucion ',fin-inicio)
    

def monitor(dataset):
    print('inicio Monitor entrenamiento')
    #global ir
    global len_training    
    count = 0
    import pandas as pd
    iron = pd.read_csv('valor_final_inicio_entrenamiento.csv')
    #en la variable starTraining se retoma el valor del índice donde terminó el último entrenamiento
    starTraining = iron.iloc[0,1]
    print('Antiguo valor de inicio: ', starTraining)
    #starTraining se sobreescribe como el valor de donde empiezan los valores válidos, diferentes de cero.
    #mientras los valores sean cero y no se haya llegado al final del dataset
    while(dataset.iloc[starTraining,[4]].values==0 and len(dataset) > starTraining+1):
        starTraining +=1     # starTraining+1
        #inicio = starTraining
        #irot = [inicio]
        df = pd.DataFrame()
        df['valor'] = [starTraining]
        df.to_csv('valor_final_inicio_entrenamiento.csv')
        
    try: 
        print('Nuevo valor de inicio ?: ', starTraining)
        inicio = starTraining
        #Mientras los valores de posición sean válidos para entrenamiento
        while(dataset.iloc[starTraining,[4]].values!=0 and dataset.iloc[starTraining,[4]].values!= ''):
                starTraining +=1 # starTraining+1
                count +=1 # count+1
                print('entra al while',count)
                #print('condicion: ',len(dataset)-5)
                #Si hay 201 valores validos seguidos, ejecuta el entrenamiento
                #if count == len(dataset)-5: #len_training:   
                if count == len(dataset)-5:                 
                    count = 0
                    print('Inicia entrenamiento en index : ',inicio , '   termina en index: ',starTraining)
                    #parametros(inicio, final, dataset)
                    ejecutar_entrenamiento(inicio,starTraining,dataset)                    
                    #guarda el valor del índice donde termina el entrenamiento
                    df = pd.DataFrame()
                    df['valor'] = [starTraining]
                    df.to_csv('valor_final_inicio_entrenamiento.csv')
                    print('Guardado valor de index donde finaliza entrenamiento')
                    read_db()
                    #Si no existen 200 valores válidos de seguido, sigue buscando en los datos.
                elif starTraining==len(dataset)-1:
                    print('ir a la funcion leer base de datos')  
                    print('No existen suficientes datos para entrenar el modelo')                      
                    read_db()
                    break                        
    except IndexError:
        print('No existen suficientes datos para entrenar el modelo')
        read_db()
    #finally:
        #   print("The 'try except' is finished")   
        #  read_db()

def read_db():
    import time
    time.sleep(5.3)
    print('inicio lectura base de datos')
    # inicialmente hace la conexion con la base de datos
    myConnection = MySQLdb.connect( host=hostname, user=username, passwd=password, db=database )    
    # genera la lectura de la base de datos
    dataset= pd.read_sql("SELECT * FROM data_set WHERE dev_id = 'tarjeta2-cubecell' order by id",myConnection)
    dataset.reset_index(inplace=True, drop=True)
    #lentrain = len(dataset)
    print('longitud del dataset: ',len(dataset))
    dataset['latitude']=dataset['latitude'].astype('float64')
    dataset['longitude']=dataset['longitude'].astype('float64')
    print('termina lectura base de datos')

    monitor(dataset)

read_db()