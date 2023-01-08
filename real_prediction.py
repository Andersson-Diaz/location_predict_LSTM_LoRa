import math
import pandas as pd
from sklearn.metrics import mean_squared_error

def haversine(lat1, lon1, lat2, lon2):
        rad=math.pi/180
        dlat=lat2-lat1
        dlon=lon2-lon1
        R=6372.795477598
        a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
        distancia=2*R*math.asin(math.sqrt(a))
        return distancia*1000

#Calculo de error en ruta 2
tarjeta2 = pd.read_csv('tarjeta2_Sandero_final.csv')
mse_lat = mean_squared_error(tarjeta2['latitude'],tarjeta2['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)
mse_lon = mean_squared_error(tarjeta2['longitude'],tarjeta2['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)
error = []
for i in range(0,len(tarjeta2)):
        error.append(haversine(tarjeta2['latitude'].iloc[i],tarjeta2['longitude'].iloc[i],tarjeta2['predicted_latitude'].iloc[i],tarjeta2['predicted_longitude'].iloc[i]))
prediccion_escenario2_tarjeta2 = pd.DataFrame()
prediccion_escenario2_tarjeta2['real_latitude'] = tarjeta2['latitude']
prediccion_escenario2_tarjeta2['real_longitude'] = tarjeta2['longitude']
prediccion_escenario2_tarjeta2['predicted_latitude'] = tarjeta2['predicted_latitude']
prediccion_escenario2_tarjeta2['predicted_longitude'] = tarjeta2['predicted_longitude']
prediccion_escenario2_tarjeta2['error[meters]'] = error
print('Error medio en tarjeta 2: ' ,haversine(0,0,mse_lat,mse_lon), 'metros')
prediccion_escenario2_tarjeta2.to_csv('real_and_predicted_position_route2.csv')

#Calculo de error en ruta 4
tarjeta4 = pd.read_csv('tarjeta4_Van.csv')
mse_lat = mean_squared_error(tarjeta4['latitude'],tarjeta4['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)
mse_lon = mean_squared_error(tarjeta4['longitude'],tarjeta4['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)
error = []
for i in range(0,len(tarjeta4)):
        error.append(haversine(tarjeta4['latitude'].iloc[i],tarjeta4['longitude'].iloc[i],tarjeta4['predicted_latitude'].iloc[i],tarjeta4['predicted_longitude'].iloc[i]))
prediccion_escenario2_tarjeta4 = pd.DataFrame()
prediccion_escenario2_tarjeta4['real_latitude'] = tarjeta4['latitude']
prediccion_escenario2_tarjeta4['real_longitude'] = tarjeta4['longitude']
prediccion_escenario2_tarjeta4['predicted_latitude'] = tarjeta4['predicted_latitude']
prediccion_escenario2_tarjeta4['predicted_longitude'] = tarjeta4['predicted_longitude']
prediccion_escenario2_tarjeta4['error[meters]'] = error
print('Error medio en tarjeta 2: ' ,haversine(0,0,mse_lat,mse_lon), 'metros')
prediccion_escenario2_tarjeta4.to_csv('real_and_predicted_position_route4.csv')

#Calculo de error en ruta 3
tarjeta5 = pd.read_csv('tarjeta5_Spark.csv')
mse_lat = mean_squared_error(tarjeta5['latitude'],tarjeta5['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)
mse_lon = mean_squared_error(tarjeta5['longitude'],tarjeta5['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)
error = []
for i in range(0,len(tarjeta5)):
        error.append(haversine(tarjeta5['latitude'].iloc[i],tarjeta5['longitude'].iloc[i],tarjeta5['predicted_latitude'].iloc[i],tarjeta5['predicted_longitude'].iloc[i]))
prediccion_escenario2_tarjeta5 = pd.DataFrame()
prediccion_escenario2_tarjeta5['real_latitude'] = tarjeta5['latitude']
prediccion_escenario2_tarjeta5['real_longitude'] = tarjeta5['longitude']
prediccion_escenario2_tarjeta5['predicted_latitude'] = tarjeta5['predicted_latitude']
prediccion_escenario2_tarjeta5['predicted_longitude'] = tarjeta5['predicted_longitude']
prediccion_escenario2_tarjeta5['error[meters]'] = error
prediccion_escenario2_tarjeta5.to_csv('real_and_predicted_position_route3.csv')
print('Error medio en tarjeta 5: ' ,haversine(0,0,mse_lat,mse_lon), 'metros')

#Calculo de error en ruta 4 CONTROLADA
tarjeta4_controlada = pd.read_csv('tarjeta4_controlada.csv')
mse_lat = mean_squared_error(tarjeta4_controlada['latitude'],tarjeta4_controlada['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)
mse_lon = mean_squared_error(tarjeta4_controlada['longitude'],tarjeta4_controlada['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)
error = []
for i in range(0,len(tarjeta4_controlada)):
        error.append(haversine(tarjeta4_controlada['latitude'].iloc[i],tarjeta4_controlada['longitude'].iloc[i],tarjeta4_controlada['predicted_latitude'].iloc[i],tarjeta4_controlada['predicted_longitude'].iloc[i]))
prediccion_escenario2_tarjeta4_controlada = pd.DataFrame()
prediccion_escenario2_tarjeta4_controlada['real_latitude'] = tarjeta4_controlada['latitude']
prediccion_escenario2_tarjeta4_controlada['real_longitude'] = tarjeta4_controlada['longitude']
prediccion_escenario2_tarjeta4_controlada['predicted_latitude'] = tarjeta4_controlada['predicted_latitude']
prediccion_escenario2_tarjeta4_controlada['predicted_longitude'] = tarjeta4_controlada['predicted_longitude']
prediccion_escenario2_tarjeta4_controlada['error[meters]'] = error
prediccion_escenario2_tarjeta4_controlada.to_csv('real_and_predicted_position_route4_controlled.csv')
print('Error medio en tarjeta 4 contolada: ' ,haversine(0,0,mse_lat,mse_lon), 'metros')

#Calculo de error en ruta 1 CONTROLADA
tarjeta1_controlada = pd.read_csv('tarjeta1_controlada.csv')
mse_lat = mean_squared_error(tarjeta1_controlada['latitude'],tarjeta1_controlada['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)
mse_lon = mean_squared_error(tarjeta1_controlada['longitude'],tarjeta1_controlada['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)
error = []
for i in range(0,len(tarjeta1_controlada)):
        error.append(haversine(tarjeta1_controlada['latitude'].iloc[i],tarjeta1_controlada['longitude'].iloc[i],tarjeta1_controlada['predicted_latitude'].iloc[i],tarjeta1_controlada['predicted_longitude'].iloc[i]))
prediccion_escenario2_tarjeta1_controlada = pd.DataFrame()
prediccion_escenario2_tarjeta1_controlada['real_latitude'] = tarjeta1_controlada['latitude']
prediccion_escenario2_tarjeta1_controlada['real_longitude'] = tarjeta1_controlada['longitude']
prediccion_escenario2_tarjeta1_controlada['predicted_latitude'] = tarjeta1_controlada['predicted_latitude']
prediccion_escenario2_tarjeta1_controlada['predicted_longitude'] = tarjeta1_controlada['predicted_longitude']
prediccion_escenario2_tarjeta1_controlada['error[meters]'] = error
prediccion_escenario2_tarjeta1_controlada.to_csv('real_and_predicted_position_route1_controlled.csv')
print('Error medio en tarjeta 1 contolada: ' ,haversine(0,0,mse_lat,mse_lon), 'metros')

#Calculo de error en ruta 5 CONTROLADA
tarjeta5_controlada = pd.read_csv('tarjeta5_controlada.csv')
mse_lat = mean_squared_error(tarjeta5_controlada['latitude'],tarjeta5_controlada['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)
mse_lon = mean_squared_error(tarjeta5_controlada['longitude'],tarjeta5_controlada['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)
error = []
for i in range(0,len(tarjeta5_controlada)):
        error.append(haversine(tarjeta5_controlada['latitude'].iloc[i],tarjeta5_controlada['longitude'].iloc[i],tarjeta5_controlada['predicted_latitude'].iloc[i],tarjeta5_controlada['predicted_longitude'].iloc[i]))
prediccion_escenario2_tarjeta5_controlada = pd.DataFrame()
prediccion_escenario2_tarjeta5_controlada['real_latitude'] = tarjeta5_controlada['latitude']
prediccion_escenario2_tarjeta5_controlada['real_longitude'] = tarjeta5_controlada['longitude']
prediccion_escenario2_tarjeta5_controlada['predicted_latitude'] = tarjeta5_controlada['predicted_latitude']
prediccion_escenario2_tarjeta5_controlada['predicted_longitude'] = tarjeta5_controlada['predicted_longitude']
prediccion_escenario2_tarjeta5_controlada['error[meters]'] = error
prediccion_escenario2_tarjeta5_controlada.to_csv('real_and_predicted_position_route5_controlled.csv')
print('Error medio en tarjeta 5 contolada: ' ,haversine(0,0,mse_lat,mse_lon), 'metros')