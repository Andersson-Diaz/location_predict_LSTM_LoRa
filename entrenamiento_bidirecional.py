
o=0
while(o==0):
    o=1
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
    #dataset.drop(index=dataset[dataset['latitude']=='0'].index, inplace=True)
    dataset.info()
    time = dataset['hour']
    dataset['latitude']=dataset['latitude'].astype('float64')
    dataset['longitude']=dataset['longitude'].astype('float64')

    inicio = 0
    ir=0
    #for i in range(inicio,inicio+200):
    while(dataset.iloc[ir,[4]].values==0):
            ir=ir+1
            inicio = ir
            print('valor de i: ', ir)
            #break

    def ejecutar_entrenamiento(r):
        print('Entrenamiento ejecutado, valor de inicio: ', r)

    for i in range(ir,ir+200):
        if dataset.iloc[i,[0]].values!=0:
            ir = ir+200
            ejecutar_entrenamiento(inicio)

        ##if dataset['latitude']!=0 and dataset['id']==200:
        ##inicio_entenamiento = 2




    window = 30
    last = int(len(dataset)/5.0)
    set_training = dataset[:-last]
    set_training.reset_index(inplace=True, drop=True)
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

    from keras.layers import Bidirectional

    modelo = Sequential()
    modelo.add(Bidirectional(LSTM(units=neurons, return_sequences=True, input_shape=dim_in)))
    modelo.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    modelo.add(Bidirectional(LSTM(50)))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=dim_out))
    modelo.compile(optimizer='Adam', loss='mse')
    modelo.fit(X_train,Y_train,epochs=10,batch_size=32)
    print(modelo.summary())
    modelo.save('model_bidirectional.h5')
    import joblib
    joblib.dump(scaler, 'scaler.save')

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
    training_distance = []
    training_distance.append(0)
    for i in range(0, len(set_training)-1):
        training_distance.append(haversine(set_training.iat[i,4],set_training.iat[i,5],set_training.iat[i+1,4],set_training.iat[i+1,5]))

    #Se crea un dataframe con la lista que contiene la distancia del conjunto de entrenamiento
    d_d_e = pd.DataFrame(training_distance, columns=['distancia'])

    #toma la hora del conjunto de datos de entrenamiento
    training_time = set_training['hour']

    #Calcula la diferencia de tiempo entre un punto de ubicacion y el anterior en el set de entrenamiento
    training_time_measure = []
    for i in range(0,len(training_time)-1):
        training_time_measure.append(training_time[i+1]-training_time[i])
    training_time_measure

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
    set_training_in_seconds.append(10)
    for i in range(0,len(training_time_measure)):
        set_training_in_seconds.append(training_time_measure[i]/delta)
    set_training_in_seconds

    #Agrupamos tiempo y distancia para usarlas como entradas del modelo de entrenamiento
    training_stack_in= np.column_stack((set_training_in_seconds,training_distance))

    #normalizamos el conjunto de datos de entrada
    sc = MinMaxScaler(feature_range=(0,1))
    set_training_time_scaled = sc.fit_transform(training_stack_in)

    # La red LSTM tendrá como entrada "window" datos consecutivos, y como salida 1 dato (la predicción se hace a
    # partir de esos "window" datos). Se conformará de esta forma el set de entrenamiento
    window_t = 30
    X_train_time = []
    Y_train_time = []

    #Se agregan las ventanas del set de entrenamiento
    for i in range(window,len(training_stack_in)):
        # X: bloques de "window" datos: 0-window, 1-window+1, 2-window+2, etc
        X_train_time.append(set_training_time_scaled[i-window:i,0:2])
        # Y: el siguiente dato
        Y_train_time.append(set_training_time_scaled[i,0:2])

    #Se pasan las listas a arrays
    X_train_time, Y_train_time = np.array(X_train_time), np.array(Y_train_time)

    # Reshape X_train para que se ajuste al modelo en Keras
    X_train_time = np.reshape(X_train_time, (X_train_time.shape[0], X_train_time.shape[1], 2))

    # Red LSTM
    #Para crear la red debemos primero definir el tamaño de los datos de entrada y del dato de salida,
    #así como el número total de neurons (neurons_time):
    dim_time_in = (X_train_time.shape[1],2)
    dim_time_out = 2
    neurons_time = 100

    #Cear un contenedor usando el módulo Sequential:
    model_time = Sequential()
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
    #Presentando a la res lstm lotess de 32 datos
    model_time.fit(X_train_time,Y_train_time,epochs=100,batch_size=32)

    #Guardamos el modelo
    model_time.save('tiempo_entrenado.h5')
    import joblib
    #guardamos el normalizador
    joblib.dump(sc, 'scaler_tiempo.save')