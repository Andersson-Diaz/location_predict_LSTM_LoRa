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
        return distancia

tarjeta2 = pd.read_csv('tarjeta2_Sandero_final.csv')
mse_lat = mean_squared_error(tarjeta2['latitude'],tarjeta2['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)

mse_lon = mean_squared_error(tarjeta2['longitude'],tarjeta2['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)

print('Error medio en tarjeta 2: ' ,haversine(0,0,mse_lat,mse_lon)*1000, 'metros')

tarjeta4 = pd.read_csv('tarjeta4_Van.csv')
mse_lat = mean_squared_error(tarjeta4['latitude'],tarjeta4['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)

mse_lon = mean_squared_error(tarjeta4['longitude'],tarjeta4['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)

print('Error medio en tarjeta 4: ' ,haversine(0,0,mse_lat,mse_lon)*1000, 'metros')