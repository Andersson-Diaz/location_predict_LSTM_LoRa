# location_predict_LSTM_LoRa
using LSTM to predicting location of buses with LoRa
the main files are:
For training: entrenamientoGeneral_tarjeta_1.py, entrenamientoGeneral_tarjeta_2.py, entrenamientoGeneral_tarjeta_4.py, entrenamientoGeneral_tarjeta_5.py.
For prediction: prediccionGeneral_tarjeta_1.py, prediccionGeneral_tarjeta_2.py, prediccionGeneral_tarjeta_4.py, prediccionGeneral_tarjeta_5.py.

The files used for training generate a trained model and a scaler to normalize and denormalize with the same scale in the prediction files. These generated files are:
Trained models(location): model_LSTM_tarjeta1.h5, model_LSTM_tarjeta2.h5, model_LSTM_tarjeta4.h5, model_LSTM_tarjeta5.h5.
Trained models (location time): tiempo_entrenado_tarjeta1.h5, tiempo_entrenado_tarjeta2.h5, tiempo_entrenado_tarjeta4.h5, tiempo_entrenado_tarjeta5.h5.
Scalers(Location): scaler_tarjeta1.save, scaler_tarjeta2.save, scaler_tarjeta4.save, scaler_tarjeta5.save.
Scalers(Time): scaler_tiempo_tarjeta1.save, scaler_tiempo_tarjeta2.save, scaler_tiempo_tarjeta4.save, scaler_tiempo_tarjeta5.save

The file inicializar.py returns to zero the database indices where training and prediction start and end.
