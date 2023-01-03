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

tarjeta2 = pd.read_csv('tarjeta2_Sandero_final.csv')
mse_lat = mean_squared_error(tarjeta2['latitude'],tarjeta2['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)

mse_lon = mean_squared_error(tarjeta2['longitude'],tarjeta2['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)

print('Error medio en tarjeta 2: ' ,haversine(0,0,mse_lat,mse_lon), 'metros')

tarjeta4 = pd.read_csv('tarjeta4_Van.csv')
mse_lat = mean_squared_error(tarjeta4['latitude'],tarjeta4['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)

mse_lon = mean_squared_error(tarjeta4['longitude'],tarjeta4['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)

print('Error medio en tarjeta 4: ' ,haversine(0,0,mse_lat,mse_lon), 'metros')

tarjeta5 = pd.read_csv('tarjeta5_Spark.csv')
mse_lat = mean_squared_error(tarjeta5['latitude'],tarjeta5['predicted_latitude'], squared=False)
print('Error en latitud: ',mse_lat)

mse_lon = mean_squared_error(tarjeta5['longitude'],tarjeta5['predicted_longitude'], squared=False)
print('Error en longitud: ',mse_lon)

print('Error medio en tarjeta 5: ' ,haversine(0,0,mse_lat,mse_lon), 'metros')

error_tarjeta2 = []
for i in range(0,len(tarjeta2)):
        error_tarjeta2.append(haversine(tarjeta2['latitude'].iloc[i],tarjeta2['longitude'].iloc[i],tarjeta2['predicted_latitude'].iloc[i],tarjeta2['predicted_longitude'].iloc[i]))

data_error_tarjeta2= pd.DataFrame(error_tarjeta2)
data_error_tarjeta2.to_csv('error_tarjeta2.csv')

error_tarjeta4 = []
for i in range(0,len(tarjeta4)):
        error_tarjeta4.append(haversine(tarjeta4['latitude'].iloc[i],tarjeta4['longitude'].iloc[i],tarjeta4['predicted_latitude'].iloc[i],tarjeta4['predicted_longitude'].iloc[i]))

data_error_tarjeta4= pd.DataFrame(error_tarjeta4)
data_error_tarjeta4.to_csv('error_tarjeta4.csv')

error_tarjeta5 = []
for i in range(0,len(tarjeta5)):
        error_tarjeta5.append(haversine(tarjeta5['latitude'].iloc[i],tarjeta5['longitude'].iloc[i],tarjeta5['predicted_latitude'].iloc[i],tarjeta5['predicted_longitude'].iloc[i]))

data_error_tarjeta5= pd.DataFrame(error_tarjeta5)
data_error_tarjeta5.to_csv('error_tarjeta5.csv')

