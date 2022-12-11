# librerias necesarias
from tokenize import Number
import numpy as np
from numpy import *
np.random.seed(4)
import tensorflow as tf
tf.random.set_seed(1)
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import time
inicio = time.time()
from datetime import timedelta
import math
import MySQLdb
import datetime
import pandas as pd
#globales
column_hour = 4
column_lat =7
column_lon = 8
column_acc = 13
column_gyro = 17
# datos para la conexion a la base de datos
hostname = '82.180.175.58'
username = 'u813407238_lora'
password = 'Seguimiento_lora_123'
database = 'u813407238_seguimiento'

#funcion para encontrar la distancia entre dos puntos geográficos
def haversine(lat1, lon1, lat2, lon2):
    rad = math.pi / 180
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6372.795477598
    a = (math.sin(rad * dlat / 2)) ** 2 + math.cos(rad * lat1) * math.cos(rad * lat2) * (math.sin(rad * dlon / 2)) ** 2
    distance = 2 * R * math.asin(math.sqrt(a))
    return distance

def ejecutar_prediccion_escenario_1(id_anterior):  # Sin LoRa
    try:
        print('inicio de ejecucion de escenario 1')
        import MySQLdb
        import pandas as pd
        import numpy as np

        #Número de datos usados para predecir una posición
        window = 30        
        # inicialmente hace la conexion con la base de datos
        mynewConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)        
        # genera la lectura de la base de datos, solo los necesario para predecir
        dataset = pd.read_sql("SELECT * FROM Tabla_General WHERE dev_id = 'tarjeta4-esp32lora' order by id DESC LIMIT 31", mynewConnection)
        mynewConnection.close()

        #Convierte los datos de posicion en flotantes           
        if 0 in dataset.latitude.values or '' in dataset.latitude.values:
            print('no hay sufucientes valores para predecir')
            mynewConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database) 

        else:
            dataset['latitude'] = dataset['latitude'].astype('float64')
            dataset['longitude'] = dataset['longitude'].astype('float64')

            #Ordena los datos de forma ascendente
            set_prediccion = pd.DataFrame()
            for i in range(0, len(dataset)):
                set_prediccion = set_prediccion.append((dataset[(len(dataset) - i - 1):(len(dataset) - i)]))
                
            #Define el tamaño de la prediccion igual al valor de la ventana 
            for i in range(window, window + window):
                set_prediccion = set_prediccion.append(dataset[window - 1:window])

            # Obtiene el valor de los indices de manera ascendente
            set_prediccion.reset_index(inplace=True, drop=True)
            #Define las características a ser ingresadas a la entrada del algoritmo de predicción
            x_test = np.column_stack((set_prediccion.iloc[1:len(set_prediccion), [column_lat]],
                                    set_prediccion.iloc[1:len(set_prediccion), [column_lon]],
                                    set_prediccion.iloc[1:len(set_prediccion), [column_acc]],
                                    set_prediccion.iloc[1:len(set_prediccion), [column_gyro]]))

            try:
                import joblib
                scaler = joblib.load('scaler_tarjeta4.save')
                #new_model = keras.models.load_model('model_bidirectional.h5')
                new_model = keras.models.load_model('model_LSTM_tarjeta4.h5')
            except OSError:
                print('No existe modelo entrenado')
                read_db()
            #normaliza el modelo
            x_test_normalized = scaler.transform(x_test)
            X_test = []
            #Da forma a los datos de entrada
            for i in range(window, len(x_test_normalized)):
                X_test.append(x_test_normalized[i - window:i, 0:5])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))           
            
            # Prediction con los primeros datos del set de validacion
            # Using predicted values to predict next step
            print('Comienza la predicción de ubicación')

            X_pred = x_test_normalized.copy()
            for i in range(window, len(X_pred)):
                xin = X_pred[i - window:i].reshape(1, window, 4)
                X_pred[i] = new_model.predict(xin)
            #Desnormaliza la predicción
            prediction = scaler.inverse_transform(X_pred)
            #Guarda la prediccion de ubicación en un archivo .csv
            calle_5_p = pd.DataFrame(prediction[:, 0:2])
            calle_5_p.to_csv('calle_5_sin_lora_prediccion.csv')
            print('fin de prediccion de ubicacion')
            ########################
            # Termina prediccion de ubicacion, empieza prediccion de tiempo
            #guarda en una serie la hora de los datos usados en predicción
            columnaTiempo = set_prediccion['hour']
            # Halla la diferencia de tiempo entre una posicion y la anterior y la guarda en una lista
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
            #divide la diferencia de tiempo entre un segundo para obtener la duracion en segundos
            validation_last = []            
            for i in range(0, len(time_validation_measured)):
                validation_last.append(time_validation_measured[i] / delta)            
            #encuentra la distancia entre una posicion y la anterior y la guarda en una lista
            distance_validation = []            
            for i in range(0, len(set_prediccion) - 1):
                distance_validation.append(
                    haversine(set_prediccion.iat[i + 1, column_lat], set_prediccion.iat[i + 1, column_lon], set_prediccion.iat[i, column_lat],
                            set_prediccion.iat[i, column_lon]))
            # agrupa la duracion y la distancia para cada punto de ubicación
            n_v = np.column_stack((validation_last, distance_validation))
            #Carga el modelo entrenado y el escalador
            try:
                import joblib
                sc = joblib.load('scaler_tiempo_tarjeta4.save')
                model_last = keras.models.load_model('tiempo_entrenado_tarjeta4.h5')
            except:
                print('El modelo no ha sido entrenado aún')
                read_db()
            #normaliza el conjunto de datos de entrada al modelo de prediccion de tiempo
            x_test_tiempo = sc.transform(n_v)
            #Reforma los datos de entrada para ajustarse al modelo.
            X_test = []
            for i in range(window, len(x_test_tiempo)):
                X_test.append(x_test_tiempo[i - window:i, 0:2])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))
            
            # Predecimos y normalizamos
            # prediction_time = model_last.predict(X_test)
            X_pred2 = x_test_tiempo.copy()
            print('Comienzo de prediccion de tiempo')

            for i in range(window, len(X_pred2)):
                xin = X_pred2[i - window:i].reshape(1, window, 2)
                X_pred2[i] = model_last.predict(xin)
            #normaliza la predicción
            prediction_time = sc.inverse_transform(X_pred2)
            print('Prediccion de tiempo:  ', prediction_time)

            #Toma un ahora de referencia como la del último dato enviado
            reference_hour = dataset['hour'].iloc[0]            
            #Obtiene la hora predicha sumando a una hora de referencia la duración predicha
            prediction_hour = []
            for i in range(window,len(prediction_time)):
                prediction_hour.append(reference_hour + timedelta(seconds=prediction_time[i,0]))
                reference_hour = reference_hour + timedelta(seconds=prediction_time[i,0])                
            #Agrupa datos de latitud, longitud y hora
            prediction_time_hour = np.column_stack((prediction[window:,0:1], prediction[window:,1:2],prediction[window:,2:3],prediction[window:,3:4],prediction_hour))
            print(prediction_time_hour)  
            pred = pd.DataFrame(prediction_time_hour[:, 0:3])
            pred.to_csv('escenario_1_predict.csv')
            #print('fin de prediccion de ubicacion')                    
            bol = True            
            #Mientras no llegue un nuevo dato o no se hayan enviado todos los datos predichos
            while(bol):                    
                from datetime import datetime
                hora_sistema = datetime.now()                
                #Halla la posicion predicha mas cercana a la hora actual
                hora_sistema = timedelta(hours=hora_sistema.hour, minutes=hora_sistema.minute, seconds=hora_sistema.second, microseconds=hora_sistema.microsecond)
                print('hora del sistema: ',hora_sistema)                
                diferencia_tiempo = []
                for i in range(0,len(prediction_hour)):
                    diferencia_tiempo.append(abs(prediction_time_hour[i,4]-hora_sistema))
                #devuelve el índice de la posición mas cercana a la hora actual
                index_actual = diferencia_tiempo.index(min(diferencia_tiempo))
                print('índex actual: ',index_actual)
                print('mínima diferencia: ',diferencia_tiempo[index_actual])                
                # genera la lectura de la base de datos
                time.sleep(1)
                mynewConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)
                dataset = pd.read_sql("SELECT * from Tabla_General WHERE dev_id = 'tarjeta4-esp32lora' order by id DESC LIMIT 1", mynewConnection)                
                #Si llega un nuevo dato desde el dispositivo LoRa, salir de la predicción
                print('id aNTERIOR: ', id_anterior)
                print('id leído: ',dataset.iloc[0,0])
                if (id_anterior!=dataset.iloc[0,0] or index_actual>=28):
                    bol = False
                    print('fin de prediccion de tiempo, llegó un nuevo dato 1')
                    break                    
                #time.sleep(1)
                #Mientras la diferencia de tiempo con la hora actual sea menor a un segundo y no se hallan enviado todos los datos predichos
                while(diferencia_tiempo[index_actual].total_seconds()<=1 and index_actual<29):                   
                    #Extrae latitud, longitud y hora predicha
                    lat = prediction_time_hour[index_actual,0]
                    lon = prediction_time_hour[index_actual,1]
                    acc= prediction_time_hour[index_actual,2]
                    giro = prediction_time_hour[index_actual,3]
                    hour = prediction_time_hour[index_actual,4]                    
                    import datetime
                    #Pasa la hora de un tipo timedelta a un tipo datetime
                    tim = datetime.timedelta(seconds=hour.total_seconds())                    
                    #Inserta en las base de datos el valor predicho correspondiente a la hora actual
                    cur = mynewConnection.cursor()
                    cadena_SQL = "INSERT INTO Tabla_General (dev_id, hour, latitude, longitude, predicted_latitude, predicted_longitude, predicted_hour, type_record,accy,gyroz) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    val = ('tarjeta4-esp32lora',tim, lat, lon, lat, lon, tim, 1, acc, giro)
                    time.sleep(0.2)
                    dataset = pd.read_sql("SELECT * from Tabla_General WHERE dev_id = 'tarjeta4-esp32lora' order by id DESC LIMIT 1", mynewConnection)
                    if (id_anterior!=dataset.iloc[0,0] or index_actual>=28): 
                        bol = False
                        print('fin de prediccion de tiempo, llegó un nuevo dato 2')
                        break  
                    cur.execute(cadena_SQL, val)
                    last_id = cur.lastrowid
                    print('Insertado en: ',last_id)
                    print("Registro creado ")
                    #incrementa el id en uno correspondiente al dato insertado
                    id_anterior = last_id                    
                    # Si no se graba, no guarda el cambio de la creacion, guarda con commit
                                        
                    mynewConnection.commit()
                    # Cierra la conexion
                    #mynewConnection.close()
                    #Borra los datos enviados de la lista de predicciones para no volverlas a enviar
                    for i in range(0,index_actual+1):
                        asd = np.delete(prediction_time_hour, i, axis=0)                    
                    #Calcula el nuevo dato predicho mas cercano a la hora actual
                    from datetime import datetime
                    hora_sistema = datetime.now()                    
                    hora_sistema = timedelta(hours=hora_sistema.hour, minutes=hora_sistema.minute, seconds=hora_sistema.second, microseconds=hora_sistema.microsecond)                    
                    diferencia_tiempo = []
                    for i in range(0,len(asd)):
                        diferencia_tiempo.append(abs(asd[i,column_hour]-hora_sistema))
                    index_actual = diferencia_tiempo.index(min(diferencia_tiempo))
                    print('index actual 2: ',index_actual)                    
                    #espera dos segundos para evitar enviar el mismo dato
                    time.sleep(1.8)
    except OSError:
            print('El modelo no ha sido entrenado aun Oserror')
            mynewConnection.close()            
    except:
            print('El modelo no ha sido entrenado aun')
            mynewConnection.close()           
    finally:
            print('fin de predicción')
            mynewConnection.close()
            read_db()
           


def ejecutar_prediccion_escenario2(ultimo_id):  # Cuando hay conexión LoRa sin GPS
    try:
        print('inicio de ejecución de escenario 2')
        import MySQLdb
        import pandas as pd
        import time
        time.sleep(2)        
        # inicialmente hace la conexion con la base de datos
        mynewConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)
        # genera la lectura de la base de datos de manera descendente para obtener los últimos valores
        dataset = pd.read_sql("SELECT * FROM Tabla_General WHERE dev_id = 'tarjeta4-esp32lora' order by id DESC", mynewConnection)       
        #Pasa los valores de posición a tipo flotante
        
        #Se define la ventana o la cantidad de datos usados para predecir el siguiente valor de posición
        long_data=30+(dataset['latitude']=='0').sum()
        dataset=dataset[0:long_data]
        dataset['latitude'] = dataset['latitude'].astype('float64')
        dataset['longitude'] = dataset['longitude'].astype('float64')
        window = 30
        # ordenamos de forma ascendente los valores obtenidos.       
        set_prediccion = pd.DataFrame()
        for i in range(0, long_data):
            set_prediccion = set_prediccion.append((dataset[(len(dataset) - i - 1):(len(dataset) - i)]))
            print(set_prediccion[i:0].values)        
        #Reinicia el indice desde cero en adelante
        set_prediccion.reset_index(inplace=True, drop=True)
        #Se forma los datos de entrada al algoritmo de prediccion con las 4 características definidas
        x_test = np.column_stack((set_prediccion.iloc[:, [column_lat]], set_prediccion.iloc[:, [column_lon]], set_prediccion.iloc[:, [column_acc]],
                                  set_prediccion.iloc[:, [column_gyro]]))    

        try:
            #Carga el modelo entrenado y el escalador de los datos de entrada
            import joblib
            scaler = joblib.load('scaler_tarjeta4.save')
            new_model = keras.models.load_model('model_LSTM_tarjeta4.h5')
            # new_model = keras.models.load_model('model_bidirectional.h5')
        except OSError:
            #captura la excepcion cuando el archivo del modelo no existe en el disco
            print('No es posible predecir, No existe modelo entrenado')
            read_db()        
        #normaliza los datos de entrada
        x_test_normalized = scaler.transform(x_test)
        #predice la posición donde el valor es cero usando los valores de IMU enviados desde el dispositivo LoRa
        X_pred = x_test_normalized.copy()
        for i in range(window, long_data):
            xin = (np.column_stack((X_pred[i - window:i, 0:2], x_test_normalized[i - window:i, 2:3],
                                    x_test_normalized[i - window:i, 3:4]))).reshape(1, window, 4)
            X_pred[i] = new_model.predict(xin)
        #desnormaliza la predicción
        prediction = scaler.inverse_transform(X_pred)
        print('Prediccion: ', prediction)
        pred = pd.DataFrame(prediction[:, 0:3])
        pred.to_csv('escenario_2_predict.csv')
        #guarda los valores predichos en un formato .csv
        calle_5_p = pd.DataFrame(prediction[:, 0:2])
        calle_5_p.to_csv('calle_5_con_lora_prediccion.csv')
        cur = mynewConnection.cursor()
        cadena_SQL = "UPDATE Tabla_General SET predicted_latitude = %s, predicted_longitude = %s, type_record = %s  WHERE id =%s"
        pr = prediction[len(prediction)-1, 0]
        t = prediction[len(prediction)-1, 1]
        val = (pr, t, 2, ultimo_id)
        cur.execute(cadena_SQL, val)
        print("Registro creado en id = {}".format(ultimo_id))
        print('tipo de dato: ',type(ultimo_id))
        # Si no se graba, no guarda el cambio de la creacion, guarda con commit
        mynewConnection.commit()
        # Cierra la conexion
        #mynewConnection.close()   
    except OSError:
            print('El modelo no ha sido entrenado aun Oserror')
            mynewConnection.close()            
    except:
            print('El modelo no ha sido entrenado aun')
            mynewConnection.close()           
    finally:
            print('fin de predicción')
            fin = time.time()
            print('tiempo de ejecucion ',fin-inicio)
            mynewConnection.close()
            read_db()

def monitor(dataset2):
    #import MySQLdb
    import pandas as pd
    from datetime import datetime
    from datetime import timedelta
    print('Comienza funcion monitor')    
    # inicialmente hace la conexion con la base de datos    
    #si el ultimo dato en la tabla no es un dato predicho en el escenario 1 o un dato nulo
    if (dataset2.iloc[0,column_lat]!= ''):
        #Lee el índice del último dato del dataset
        ultimo_id = dataset2.iloc[0, 0]
        #Lee el índice donde terminó la anterior prediccion.
        iron = pd.read_csv('id_prediccion_tarjeta4.csv')
        id_guardado = iron.iloc[0, 1]

        # guarda la diferencia de tiempo en segundos desde la hora actual hasta la hora del ultimo registro recibido.
        #Obtiene la hora del sistema
        hora_sistema = datetime.now()
        #Pasa la hora del sistema a un formato timedelta (duración en segundos)
        hora_sistema = timedelta(hours=hora_sistema.hour, minutes=hora_sistema.minute, seconds=hora_sistema.second,
                                microseconds=hora_sistema.microsecond)
        print('hora actual del sistema: ',hora_sistema)
        #Se crea un objeto timedelta de duración de 1 segundo
        delta = timedelta(
            days=0,
            seconds=1,
            microseconds=0,
            milliseconds=0,
            minutes=0,
            hours=0,
            weeks=0)
        # En s se guarda la diferencia de tiempo entre la hora actual y la hora del último registro
        # Se divide entre delta para que el resultado se exprese en numero de segundos.
        s = abs(dataset2.iloc[0, column_hour] - hora_sistema) / delta
        print(s)

        #  si llegó un nuevo dato
        print(ultimo_id)
        print(id_guardado)
        if ultimo_id != id_guardado:            
            #Se crea un Dataframe para guardar el valor del índice del último dato evaluado
            df = pd.DataFrame()
            df['valor'] = [dataset2.iloc[0, 0]]
            df.to_csv('id_prediccion_tarjeta4.csv')
            print('Guardar valor de id prediccion')
            #si el dato de posición que llegó es igual a cero, ejecutar prediccion escenario 2
            print('Valor medido de id', dataset2.iloc[0, 0])
            if dataset2.iloc[0, column_lat] == '0':
                print('Dato nuevo es igual a cero')
                #se pasa como parámetro, el índice del nuevo dato igual a cero
                ejecutar_prediccion_escenario2(ultimo_id)
            else:
                #Si el nuevo dato no es igual a cero, entonces vuelve a leer la base de datos para ver si llegó un nuevo dato
                print('Inicio monitorizacion')
                read_db()
            # ¿Ha pasado mas de 13 segundos desde el ultimo dato enviado?
        elif s > (13):
            #último_id es el índice desde donde debe el algoritmo contar los valores para predecir
            ejecutar_prediccion_escenario_1(ultimo_id)
        else:
            print('No es necesario predecir')
            time.sleep(4)
            read_db()
    else:
        print('último dato inválido, esperando nuevos datos')
        time.sleep(3)
        read_db()

def read_db():
    import time
    time.sleep(5.3)
    print('inicio lectura base de datos')
    # inicialmente hace la conexion con la base de datos
    myConnection = MySQLdb.connect( host=hostname, user=username, passwd=password, db=database )    
    # genera la lectura de la base de datos
    dataset= pd.read_sql("SELECT * FROM Tabla_General WHERE dev_id = 'tarjeta4-esp32lora' order by id DESC LIMIT 1",myConnection)
    myConnection.close()    
    #Pasa los valores de posicion a tipo flotante
    """dataset['latitude']=dataset['latitude'].astype('float64')
    dataset['longitude']=dataset['longitude'].astype('float64')
    dataset['id'] = dataset['id'].astype('float64')"""
    print('termina lectura base de datos')
    monitor(dataset)

#funcion que dispara la ejecucion de todo el algoritmo
read_db()
