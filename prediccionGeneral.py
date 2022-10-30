# librerias necesarias
from tokenize import Number
import numpy as np
from numpy import *

np.random.seed(4)
import matplotlib.pyplot as plt
# import pandas as pd
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout

import time
from datetime import datetime
from datetime import timedelta
import math
import MySQLdb


# import pandas as pd

# dataset2= pd.read_sql("SELECT * FROM LoRaWAN_messages_calle_5 order by id ASC lIMIT 1",myConnection)
def haversine(lat1, lon1, lat2, lon2):
    rad = math.pi / 180
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6372.795477598
    a = (math.sin(rad * dlat / 2)) ** 2 + math.cos(rad * lat1) * math.cos(rad * lat2) * (math.sin(rad * dlon / 2)) ** 2
    distance = 2 * R * math.asin(math.sqrt(a))
    return distance


def ejecutar_prediccion_escenario_1(idx):  # Sin LoRa
    try:
        print('inicio de ejecucion de escenario 1')
        import MySQLdb
        import pandas as pd
        import numpy as np

        window = 30
        # datos para la conexion a la base de datos
        hostname = '82.180.175.58'
        username = 'u813407238_lora'
        password = 'Seguimiento_lora_123'
        database = 'u813407238_seguimiento'
        # inicialmente hace la conexion con la base de datos
        myConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)

        # genera la lectura de la base de datos
        dataset = pd.read_sql("SELECT * FROM LoRaWAN_messages order by id DESC LIMIT 31", myConnection)
        dataset.drop(index=dataset[dataset['latitude'] == '0'].index, inplace=True)
        print("Va a imprimir el dataset leido de la BD...")
        # dataset.drop(index=dataset[dataset['latitude']=='0'].index, inplace=True)
        dataset.info()
        # time = dataset['hour']
        dataset['latitude'] = dataset['latitude'].astype('float64')
        dataset['longitude'] = dataset['longitude'].astype('float64')

        set_prediccion = pd.DataFrame()
        for i in range(0, len(dataset)):
            set_prediccion = set_prediccion.append((dataset[(len(dataset) - i - 1):(len(dataset) - i)]))
            print(set_prediccion[i:0].values)

        for i in range(window, window + window):
            set_prediccion = set_prediccion.append(dataset[window - 1:window])

        print(set_prediccion)

        # Obtenemos
        set_prediccion.reset_index(inplace=True, drop=True)
        # x_test_latitud = set_prediccion.iloc[:,4:5]
        x_test = np.column_stack((set_prediccion.iloc[1:len(set_prediccion), [4]],
                                  set_prediccion.iloc[1:len(set_prediccion), [5]],
                                  set_prediccion.iloc[1:len(set_prediccion), [8]],
                                  set_prediccion.iloc[1:len(set_prediccion), [11]]))

        array_latitud = []
        for x in range(len(x_test)):
            array_latitud.append(x_test[x, 0])

        array_longitud = []
        for x in range(len(x_test)):
            array_longitud.append(x_test[x, 1])

        import joblib
        scaler = joblib.load('scaler.save')

        x_test_normalized = scaler.transform(x_test)

        X_test = []
        for i in range(window, len(x_test_normalized)):
            X_test.append(x_test_normalized[i - window:i, 0:5])

        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

        new_model = keras.models.load_model('model_LSTM.h5')

        # new_model = keras.models.load_model('model_bidirectional.h5')

        def haversine(lat1, lon1, lat2, lon2):
            rad = math.pi / 180
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            R = 6372.795477598
            a = (math.sin(rad * dlat / 2)) ** 2 + math.cos(rad * lat1) * math.cos(rad * lat2) * (
                math.sin(rad * dlon / 2)) ** 2
            distance = 2 * R * math.asin(math.sqrt(a))
            return distance

        # Prediction con los primeros datos del set de validacion
        # Using predicted values to predict next step
        print('Comienzo de prediccion de ubicacion')

        X_pred = x_test_normalized.copy()
        for i in range(window, len(X_pred)):
            xin = X_pred[i - window:i].reshape(1, window, 4)
            X_pred[i] = new_model.predict(xin)

        prediction = scaler.inverse_transform(X_pred)
        calle_5_p = pd.DataFrame(prediction[:, 0:2])
        calle_5_p.to_csv('calle_5_sin_lora_prediccion.csv')
        print('fin de prediccion de ubicacion')
        ########################
        # Termina prediccion de ubicacion, empieza prediccion de tiempo

        columnaTiempo = set_prediccion['hour']
        # Hallar la diferencia de tiempo entre una posicion y la anterior
        time_validation_measured = []
        for i in range(0, len(columnaTiempo) - 1):
            time_validation_measured.append(columnaTiempo[i + 1] - columnaTiempo[i])

        # Creamos un objeto deltatime de value 1 segundo
        # Al dividir deltatime / deltatime se obtiene un value de tipo float
        # Al dividir sobre un segundo se obtiene un value de tiempo en segundos
        delta = timedelta(
            days=0,
            seconds=1,
            microseconds=0,
            milliseconds=0,
            minutes=0,
            hours=0,
            weeks=0)

        validation_last = []
        # validation_last.append(10)
        for i in range(0, len(time_validation_measured)):
            validation_last.append(time_validation_measured[i] / delta)

        validation_last
        last_data_validation = pd.DataFrame(validation_last, columns=['duracion'])
        last_data_validation

        # x_test = last_data_validation.values

        def haversine(lat1, lon1, lat2, lon2):
            rad = math.pi / 180
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            R = 6372.795477598
            a = (math.sin(rad * dlat / 2)) ** 2 + math.cos(rad * lat1) * math.cos(rad * lat2) * (
                math.sin(rad * dlon / 2)) ** 2
            distance = 2 * R * math.asin(math.sqrt(a))
            return distance

        distance_validation = []
        # distance_validation.append(0)
        for i in range(0, len(set_prediccion) - 1):
            distance_validation.append(
                haversine(set_prediccion.iat[i + 1, 4], set_prediccion.iat[i + 1, 5], set_prediccion.iat[i, 4],
                          set_prediccion.iat[i, 5]))
        n_v = np.column_stack((validation_last, distance_validation))
        import joblib
        sc = joblib.load('scaler_tiempo.save')
        x_test_tiempo = sc.transform(n_v)

        X_test = []
        for i in range(window, len(x_test_tiempo)):
            X_test.append(x_test_tiempo[i - window:i, 0:2])

        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))

        model_last = keras.models.load_model('tiempo_entrenado.h5')
        # Predecimos y normalizamos
        # prediction_time = model_last.predict(X_test)
        X_pred2 = x_test_tiempo.copy()
        print('Comienzo de prediccion de tiempo')

        for i in range(window, len(X_pred2)):
            xin = X_pred2[i - window:i].reshape(1, window, 2)
            X_pred2[i] = model_last.predict(xin)
        prediction_time = sc.inverse_transform(X_pred2)
        print('Prediccion de tiempo:  ', prediction_time)
        reference_hour = dataset['hour'].iloc[0]
        #prediction_hour = set_prediccion['hour'].copy()
        prediction_hour = []
        for i in range(window,len(prediction_time)):
            prediction_hour.append(reference_hour + timedelta(seconds=prediction_time[i,0]))
            reference_hour = reference_hour + timedelta(seconds=prediction_time[i,0])
            #print("suma tiempo: ",prediction_hour[i-window+1])
            #print(prediction_hour[i])
            #hora = prediction_hour[i]
            #print("hora:")
            #print(hora)
        prediction_time_hour = np.column_stack((prediction[window:,0:1], prediction[window:,1:2],prediction_hour))
        print(prediction_time_hour)

        hora_sistema = datetime.now()
        print(hora_sistema)

        hora_sistema = timedelta(hours=hora_sistema.hour, minutes=hora_sistema.minute, seconds=hora_sistema.second, microseconds=hora_sistema.microsecond)
        print(hora_sistema)

        diferencia_tiempo = []
        for i in range(0,len(prediction_hour)):
            diferencia_tiempo.append(abs(hora_sistema - prediction_hour[i]))

        print(diferencia_tiempo)

        index = diferencia_tiempo.index(min(diferencia_tiempo))
        print(index)
        lat = prediction_time_hour[index,0]
        lon = prediction_time_hour[index,1]
        hour = prediction_time_hour[index,2]

        mynewConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)
        cur = mynewConnection.cursor()
        cadena_SQL = "INSERT INTO Tabla_General (latitude_p_e_1, longitude_p_e_1, hour_p, type_record) VALUES(%s,%s,%s,%s)"
        val = (lat, lon, hour, 1)
        cur.execute(cadena_SQL, val)
        print("Registro creado ")
        #print('tipo de dato: ',type(idx))
        # Si no se graba, no guarda el cambio de la creacion, guarda con commit
        mynewConnection.commit()
        # Cierra la conexion
        mynewConnection.close()


        print('fin de prediccion de tiempo')
    except OSError:
        print('El modelo no ha sido entrenado aun')
        monitor()
    except:
        print('El modelo no ha sido entrenado aun')
        monitor()
    finally:
        print('finally')
        monitor()


def ejecutar_prediccion_escenario2(idx):  # Cuando hay conexión LoRa sin GPS
    try:
        print('inicio de ejecución de escenario 2')
        import MySQLdb
        import pandas as pd
        import time
        time.sleep(2)
        #idx=40
        # datos para la conexion a la base de datos
        hostname = '82.180.175.58'
        username = 'u813407238_lora'
        password = 'Seguimiento_lora_123'
        database = 'u813407238_seguimiento'
        # inicialmente hace la conexion con la base de datos
        myConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)

        # genera la lectura de la base de datos
        dataset = pd.read_sql("SELECT * FROM LoRaWAN_messages order by id DESC LIMIT 31", myConnection)

        print("Va a imprimir el dataset leido de la BD...")
        # dataset.drop(index=dataset[dataset['latitude']=='0'].index, inplace=True)
        dataset.info()
        # time = dataset['hour']
        dataset['latitude'] = dataset['latitude'].astype('float64')
        dataset['longitude'] = dataset['longitude'].astype('float64')
        window = 30
        # Obtenemos
        # last = int(len(dataset)/5.0)
        # set_prediccion = dataset
        set_prediccion = pd.DataFrame()
        for i in range(0, len(dataset)):
            set_prediccion = set_prediccion.append((dataset[(len(dataset) - i - 1):(len(dataset) - i)]))
            print(set_prediccion[i:0].values)

        # set_prediccion = dataset[-last-window:]
        set_prediccion.reset_index(inplace=True, drop=True)
        # x_test_latitud = set_prediccion.iloc[:,4:5]
        x_test = np.column_stack((set_prediccion.iloc[:, [4]], set_prediccion.iloc[:, [5]], set_prediccion.iloc[:, [8]],
                                  set_prediccion.iloc[:, [11]]))

        array_latitud = []
        for x in range(len(x_test)):
            array_latitud.append(x_test[x, 0])

        array_longitud = []
        for x in range(len(x_test)):
            array_longitud.append(x_test[x, 1])

        try:
            new_model = keras.models.load_model('model_LSTM.h5')
            # new_model = keras.models.load_model('model_bidirectional.h5')
        except OSError:
            print('No es posible predecir, No existe modelo entrenado')
            monitor()

        import joblib
        scaler = joblib.load('scaler.save')

        x_test_normalized = scaler.transform(x_test)

        X_pred = x_test_normalized.copy()
        for i in range(window, len(X_pred)):
            xin = (np.column_stack((X_pred[i - window:i, 0:2], x_test_normalized[i - window:i, 2:3],
                                    x_test_normalized[i - window:i, 3:4]))).reshape(1, window, 4)
            X_pred[i] = new_model.predict(xin)

        prediction = scaler.inverse_transform(X_pred)
        print('Prediccion: ', prediction)
        print(prediction)
        calle_5_p = pd.DataFrame(prediction[:, 0:2])
        calle_5_p.to_csv('calle_5_con_lora_prediccion.csv')
        mynewConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)
        cur = mynewConnection.cursor()
        cadena_SQL = "UPDATE Tabla_General SET latitude_p_e_2 = %s, longitude_p_e_2 = %s, type_record = %s  WHERE id =%s"
        pr = prediction[window, 0]
        t = prediction[window, 1]
        idx
        val = (pr, t, 2, idx)
        cur.execute(cadena_SQL, val)
        print("Registro creado en id = {}".format(idx))
        print('tipo de dato: ',type(idx))
        # Si no se graba, no guarda el cambio de la creacion, guarda con commit
        mynewConnection.commit()
        # Cierra la conexion
        mynewConnection.close()
        # monitor()
    except OSError:
        print('El modelo no ha sido entrenado aun')
        monitor()
    except:
        print('El modelo no ha sido entrenado aun')
        monitor()
    finally:
        print('finally')
        monitor()


def monitor():
    import MySQLdb
    import pandas as pd
    from datetime import datetime
    from datetime import timedelta
    print('Comienza funcion monitor')
    # datos para la conexion a la base de datos
    hostname = '82.180.175.58'
    username = 'u813407238_lora'
    password = 'Seguimiento_lora_123'
    database = 'u813407238_seguimiento'

    # inicialmente hace la conexion con la base de datos
    myConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)

    # genera la lectura de la base de datos
    dataset2 = pd.read_sql("SELECT * FROM LoRaWAN_messages order by id DESC LIMIT 1", myConnection)
    dataset2['id'] = dataset2['id'].astype('float64')
    idx = dataset2.iloc[0, 0]
    idx.astype('float64')
    iron = pd.read_csv('id_prediccion.csv')
    id2 = iron.iloc[0, 1]

    # guarda la diferencia de tiempo en segundos desde la hora actual hasta la hora del ultimo registro recibido.
    hora_sistema = datetime.now()
    print(dataset2['hour'])
    hora_sistema = timedelta(hours=hora_sistema.hour, minutes=hora_sistema.minute, seconds=hora_sistema.second,
                             microseconds=hora_sistema.microsecond)
    print(hora_sistema)
    print(abs(dataset2['hour'] - hora_sistema))
    delta = timedelta(
        days=0,
        seconds=1,
        microseconds=0,
        milliseconds=0,
        minutes=0,
        hours=0,
        weeks=0)
    # En s se guarda la diferencia de tiempo
    s = abs(dataset2['hour'] - hora_sistema) / delta
    print(s)

    # Llegó un nuevo dato?
    if idx != id2:
        print("Va a imprimir el dataset leido de la BD...")
        # dataset.drop(index=dataset[dataset['latitude']=='0'].index, inplace=True)
        dataset2.info()
        # time = dataset['hour']
        dataset2['latitude'] = dataset2['latitude'].astype('float64')
        dataset2['longitude'] = dataset2['longitude'].astype('float64')
        iro = dataset2.iloc[0, [0]].values
        df = pd.DataFrame()
        df['valor'] = iro
        df.to_csv('id_prediccion.csv')
        print('Guardar valor de id prediccion')

        # dataset2= pd.read_sql("SELECT * FROM LoRaWAN_messages_calle_5 order by id ASC lIMIT 1",myConnection)
        print('Valor medido de id', dataset2.iloc[0, [0]].values)
        if dataset2.iloc[0, [4]].values == 0:
            print('inicia condicional')
            ejecutar_prediccion_escenario2(idx)
        else:
            print('Inicio monitorizacion')
            monitor()
    # ¿Ha tardado un dato en llegar mas de 13 segundos?
    elif s.values > (13):
        ejecutar_prediccion_escenario_1(idx)
    else:
        print('No es necesario predecir')
        time.sleep(5)
        monitor()


monitor()
