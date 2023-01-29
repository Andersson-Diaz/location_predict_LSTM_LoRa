# location_predict_LSTM_LoRa
using LSTM to predicting location of buses with LoRa
los principales archivos son:
Para entrenameinto: entrenamientoGeneral_tarjeta_1.py, entrenamientoGeneral_tarjeta_2.py, entrenamientoGeneral_tarjeta_4.py, entrenamientoGeneral_tarjeta_5.py.
Para prediccion: prediccionGeneral_tarjeta_1.py, prediccionGeneral_tarjeta_2.py, prediccionGeneral_tarjeta_4.py, prediccionGeneral_tarjeta_5.py.

Los archivos usados para entrenamiento generan un modelo entrenado y un escalador para normalizar y desnormalizar con la misma escala en los archivos de prediccion. 
Estos archivos generados son:
Modelos entrenados(ubicacion): model_LSTM_tarjeta1.h5, model_LSTM_tarjeta2.h5, model_LSTM_tarjeta4.h5, model_LSTM_tarjeta5.h5.
Modelos entrenados(tiempo de ubicacion): tiempo_entrenado_tarjeta1.h5, tiempo_entrenado_tarjeta2.h5, tiempo_entrenado_tarjeta4.h5, tiempo_entrenado_tarjeta5.h5.
Escaladores(Ubicacion): scaler_tarjeta1.save, scaler_tarjeta2.save, scaler_tarjeta4.save, scaler_tarjeta5.save.
Escaladores(Tiempo): scaler_tiempo_tarjeta1.save, scaler_tiempo_tarjeta2.save, scaler_tiempo_tarjeta4.save, scaler_tiempo_tarjeta5.save

El archivo inicializar.py devuelve a cero los indices de la base de datos donde empieza y termina el entrenamiento  y la prediccion.
