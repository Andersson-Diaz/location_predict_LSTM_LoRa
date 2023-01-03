
#Archivo para inicializar los indices de inicio de prediccion de las tarjetas LoRa
import pandas as pd
df = pd.DataFrame()
df['valor'] = [1]
df.to_csv('id_prediccion_tarjeta1.csv')

import pandas as pd
df = pd.DataFrame()
df['valor'] = [1]
df.to_csv('id_prediccion_tarjeta2.csv')

import pandas as pd
df = pd.DataFrame()
df['valor'] = [1]
df.to_csv('id_prediccion_tarjeta4.csv')

import pandas as pd
df = pd.DataFrame()
df['valor'] = [1]
df.to_csv('id_prediccion_tarjeta5.csv')

import pandas as pd
df2 = pd.DataFrame()
df2['valor'] = [1]
df2.to_csv('valor_final_inicio_entrenamiento.csv')

print('índices inicializados')