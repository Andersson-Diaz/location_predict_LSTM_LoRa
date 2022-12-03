
import pandas as pd
df = pd.DataFrame()
df['valor'] = [1]
df.to_csv('id_prediccion.csv')

df2 = pd.DataFrame()
df2['valor'] = [1]
df2.to_csv('valor_final_inicio_entrenamiento.csv')

print('índices inicializados')
import sys
sys.maxsize