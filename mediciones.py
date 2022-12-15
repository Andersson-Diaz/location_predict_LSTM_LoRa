
import pandas as pd
import MySQLdb
# datos para la conexion a la base de datos
hostname = '82.180.175.58'
username = 'u813407238_lora'
password = 'Seguimiento_lora_123'
database = 'u813407238_seguimiento'
myConnection = MySQLdb.connect( host=hostname, user=username, passwd=password, db=database )    
# genera la lectura de la base de datos
dataset= pd.read_sql("SELECT * FROM Tabla_General WHERE dev_id = 'tarjeta4-esp32lora' AND type_record = 1 order by id ASC",myConnection)
myConnection.close()    
dataset.to_csv('tarjeta4_Van.csv')

myConnection = MySQLdb.connect( host=hostname, user=username, passwd=password, db=database )    
# genera la lectura de la base de datos
dataset= pd.read_sql("SELECT * FROM Tabla_General WHERE dev_id = 'tarjeta2-cubecell' AND type_record = 1 order by id ASC",myConnection)
myConnection.close()    
dataset.to_csv('tarjeta2_Sandero.csv')

myConnection = MySQLdb.connect( host=hostname, user=username, passwd=password, db=database )    
# genera la lectura de la base de datos
dataset= pd.read_sql("SELECT * FROM Tabla_General WHERE dev_id = 'tarjeta5-cubecell' AND type_record = 1 order by id ASC",myConnection)
myConnection.close()    
dataset.to_csv('tarjeta5_Spark.csv')