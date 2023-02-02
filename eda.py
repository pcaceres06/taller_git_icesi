# -*- coding: utf-8 -*-
import os
import re
import json
from datetime import datetime

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mi
import pycircular as pc
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils.grahps import circular_graph, biplot

pd.options.display.max_columns = None
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.2f}'.format

FILE_PATH = f'../../data/raw'

# Cargar datos
df = pd.read_csv(f'{FILE_PATH}/DS_Challenge.csv', sep=';')

"""
Se realiza el cargue inicial del dataset sin realizar parseo de datos.

El dataset consta de de 16 columns y 26975 registros.

Variables
----------
ID_USER: id del usuario
genero: masculino
monto: valor de la transaccion
fecha: fecha de la transaccion
hora: hora de la transaccion
dispositivo: json con modelo, score y so del dispositivo
establecimiento
ciudad
tipo_tc: tipo tarjeta de credito (fisica/virtual)
linea_tc: codigo bin?
interes_tc: ??
status_txn: estado de la transacción
is_prime: ??
dcto: porcentaje descuento
cashback: porcentaje cashback
fraude: variable objetivo
"""

"""
PREGUNTAR
--------------
- A que hace referencia la variable is_prime
- a que hace referencia los valores de device_score
- La linea_tc es el bin??
- a que hace referencia los valores interes_tc
- el establecimiento es el lugar donde se hizo la transacción?
- la ciudad es la ciudad donde se hizo la transacción?
"""

"""
Al cargar el dataset se observan inicialmente errores en el formato

Errores al cargar los datos
-- Variable genero con valores '--' (se considera missing)
-- Valores decimales con ',' como separador decimal (puede deberse al idioma
del equipo donde se generó la data)
-- Estructura json de la variable dispositivo con error, utiliza como separador
de llavel ';' en lugar de ',' y las llaves no están encerradas entre comillas
dobles
"""

#==========================================
# Limpieza inicial de los datos
#==========================================

# normalizar titulos encabezado
df.columns = df.columns.str.lower()

# Ajustar formato fecha
df.fecha = pd.to_datetime(df.fecha, dayfirst=True)

# Ajustar valores decimales
df.monto = df.monto.str.replace(',', '.').astype(float)
df.dcto = df.dcto.str.replace(',', '.').astype(float)
df.cashback = df.cashback.str.replace(',', '.').astype(float)

# Tipo tarjeta
df.loc[df.tipo_tc == 'FÃ­sica', 'tipo_tc'] = 'Fisica'

# Genero
df.loc[df.genero == '--', 'genero'] = np.nan

# Dispositivo
df.dispositivo = df.dispositivo.str.replace(';', ',')
df.dispositivo = df.dispositivo.str.replace("'", "\"")
df[['device_model', 
    'device_score', 
    'device_os']] = (pd.json_normalize([json.loads(d) 
                                        for d in df.dispositivo]))

df.loc[df.device_os.isin(['%%', ',']), 'device_os'] = np.nan


# Reordenar columnas
df = df[['id_user', 'genero', 'monto', 'fecha', 'hora',
         'device_model', 'device_score', 'device_os', 'establecimiento', 
         'ciudad', 'tipo_tc', 'linea_tc', 'interes_tc',
         'status_txn', 'is_prime', 'dcto', 'cashback', 'fraude']]


# Convertir variables object a category
cat_cols = ['genero', 'device_score',
            'device_os', 'establecimiento', 
            'ciudad', 'tipo_tc','status_txn']
df[cat_cols] = df[cat_cols].astype('category')

# Converitr bool a int
df.fraude = df.fraude * 1
df.fraude = df.fraude.astype('int')

df.is_prime = df.is_prime * 1
df.is_prime = df.is_prime.astype('int')

#==========================================
# Missing Data
#==========================================
mi.bar(df)
plt.show()

pct_missing = pd.DataFrame(df.isna().sum() / len(df), columns=['porcentaje'])
pct_missing.plot(kind='bar',
                 title='Porcentaje datos faltantes')

"""
Dada que las variables device_os, establecimiento y ciudad tienen un 
porcentaje de datos faltantes muy alto, se decide excluir estas variables
para la creación del modelo.

(que significan los nulos)


La variable genero tiene un 10% de datos faltantes, se considerará
imputar la variable
"""

#==========================================
# Creacion de variables
#==========================================
df['fecha_hora'] = [
                    datetime.strptime(f"{df.fecha[row].strftime('%Y-%m-%d')} {df.hora[row]}:00:00",
                                      '%Y-%m-%d %H:%M:%S')
                    for row in range(0, len(df))
                    ]

# franquicia_tc = {1:'aerolinea',
#                  2:'aerolinea',
#                  3:'american',
#                  4: 'visa',
#                  5: 'master',
#                  6: 'discovery',
#                  7: 'petrolera',
#                  8: 'telcos',
#                  9: 'pais_ope'}

# df.linea_tc = df.linea_tc.astype(str)
# df['franquicia_tc'] = df.linea_tc.apply(lambda x: int(x[0]))
# df['franquicia_tc'] = df['franquicia_tc'].map(franquicia_tc)
# df['franquicia_tc'] = df['franquicia_tc'].astype('category')

# Tipo de dia
df['dia_semana'] = df.fecha.dt.day_of_week
df['dia_mes'] = df.fecha.dt.day

# Partes del dia
df.loc[(df.hora >= 0) & (df.hora < 5), 'parte_dia'] = 'madrugada'
df.loc[(df.hora >= 5) & (df.hora < 12), 'parte_dia'] = 'mañana'
df.loc[(df.hora >= 12) & (df.hora < 15), 'parte_dia'] = 'mediodia'
df.loc[(df.hora >= 15) & (df.hora < 20), 'parte_dia'] = 'tarde'
df.loc[(df.hora >= 20) & (df.hora <= 23), 'parte_dia'] = 'noche'
df.parte_dia = df.parte_dia.astype('category')

# Radianes
df['radian'] = pc.utils._date2rad(df.fecha_hora, time_segment='hour')

# Monto neto (validar con ana)
df['monto_neto'] = df.monto - df.cashback - df.dcto

# Porcentaje cupo
df['pct_cupo'] = df.monto / df.linea_tc

#=======================================================
# Analisis Exploratorio de datos
#=======================================================

#============================================
# Analisis Univariado
#============================================

# Ordenar datos por usuario y timestamp
df = df.sort_values(by=['id_user', 'fecha_hora'],
                    ascending=[True, True])

df = df.reset_index(drop=True)


# Distribucion de la variable objetivo
target_var = df.fraude
target_var.value_counts() / len(df)

""" La variable objetivo se encuentra desbalanceada, con un 97% de 
transacciones que no son fraudes y 3% que SI son franudes
"""

# Analisis de variables categoricas
cat_df = df.select_dtypes(include=['category', 'object'])
cat_df['fraude'] = target_var


fig, axes = plt.subplots(3, 3, figsize=(20, 15))
sns.barplot(data=cat_df, x='genero', y='fraude', ax=axes[0, 0])
sns.barplot(data=cat_df, x='device_os', y='fraude', ax=axes[0, 1])
sns.barplot(data=cat_df, x='establecimiento', y='fraude',  ax=axes[0, 2])
sns.barplot(data=cat_df, x='ciudad', y='fraude', ax=axes[1, 0])
sns.barplot(data=cat_df, x='tipo_tc', y='fraude', ax=axes[1, 1])
sns.barplot(data=cat_df, x='status_txn', y='fraude', ax=axes[1, 2])
sns.barplot(data=cat_df, x='device_score', y='fraude', ax=axes[2, 0])
sns.barplot(data=cat_df, x='parte_dia', y='fraude', ax=axes[2, 1])
plt.show()

(
    pd.
    crosstab(index=cat_df.ciudad, 
             columns=cat_df.fraude, 
             margins=False,
             normalize='index')
    .plot(kind='bar', stacked=True)
)


""" Entre las variables categórica no pareciera haber una diferencia
entre las transacciones de fraude y las que no lo son
"""

# Analisis de variables cuantitativas
num_df = df.select_dtypes(exclude=['category', 'object'])

(
    num_df[['monto', 'hora', 'device_model', 'linea_tc', 'interes_tc',
        'is_prime', 'dcto', 'cashback', 'dia_semana', 
        'dia_mes', 'radian', 'monto_neto', 'pct_cupo',
        'fraude']]
    .describe()
    .T
 )

# Matriz de correlaciones
corr_mat = num_df[['monto', 'hora', 'linea_tc', 'interes_tc', 
                   'dcto', 'cashback', 'dia_semana', 
                   'dia_mes', 'radian', 'is_prime',
                   'monto_neto', 'pct_cupo',
                   'fraude']].corr(method='pearson')
lower_mat = np.triu(corr_mat, k=1)

plt.figure(figsize=(15, 8))
sns.heatmap(corr_mat,
            annot=True, 
            vmin=-1, 
            vmax=1,
            #cmap=sns.diverging_palette(20, 220, n=200),
            #cmap=sns.diverging_palette(220, 20, as_cmap=True),
            #cmap=sns.color_palette('Blues'),
            cmap=sns.color_palette('viridis'),
            mask=lower_mat
           )
plt.show()

"""
- Se observa una correlación fuerte entre la variable cashback y el monto.
Igualment se observa una correlación moderada entre las variables monto
y dcto

- La variable fraude tiene una correlacion debil directa con las variables
hora y dia_mes; y una correlacion inversa debil con la variable
dia_semana, linea_tc y el monto
"""

# Boxplots
num_cols = ['monto', 'hora', 'interes_tc', 'dcto', 
            'cashback', 'dia_semana', 'dia_mes',
            'linea_tc', 'monto_neto', 'pct_cupo']


plt.figure(figsize=(15, 20))
i = 1
for col in num_cols:
    plt.subplot(4, 3, i)
    sns.boxplot(data=num_df, y=col)
    plt.title(f"Variable - {col}")
    i+=1

plt.figure(figsize=(15, 20))
i = 1
for col in num_cols:
    plt.subplot(4, 3, i)
    sns.boxplot(data=num_df, x='fraude', y=col)
    plt.title(f"Variable - {col}")
    i+=1

# Cantidad de transacciones por hora
hour_txn = (
            df
            .groupby(by=['hora', 'fraude'])
            .count()['id_user']
            .reset_index()
            )

plt.figure(figsize=(15, 10))
sns.barplot(data=hour_txn, x='hora', y='id_user', hue='fraude')
plt.show()

#================================================
# Analisis periodico de horas de transacciones
#================================================

# Analisis periodico de horas de transacciones
circular_graph(data=df,
               title='Distribucion transacciones por hora',
               time_segment='hour')

# Transacciones normales
normal_txn = df.query('fraude == 0').copy()
circular_graph(data=normal_txn,  
               title='Distribucion transacciones normales por hora',
               time_segment='hour')

# Transacciones fraudulentas
fraude_txn = df.query('fraude == 1').copy()
circular_graph(data=fraude_txn, 
               title='Distribucion transacciones fraudulentas por hora',
               time_segment='hour')

#=======================================
# Transacciones por dia de la semana
#=======================================
circular_graph(data=df,  
               title='Distribución transacciones por dia de la semana',
               time_segment='dayweek')

# Transacciones normales
normal_txn = df.query('fraude == 0').copy()
circular_graph(normal_txn, 
               title='Distribución transacciones normales por dia de la semana',
               time_segment='dayweek')

# Transacciones fraudulentas
fraude_txn = df.query('fraude == 1').copy()
circular_graph(fraude_txn, 
               title='Distribución transacciones fraudulentas por dia de la semana',
               time_segment='dayweek')

#=======================================
# Transacciones por dia de la semana
#=======================================
circular_graph(data=df,
               title='Distribución transacciones por dia del mes',
               time_segment='daymonth')

# Transacciones normales
normal_txn = df.query('fraude == 0').copy()
circular_graph(normal_txn, 
               title='Distribución transacciones normales por dia del mes',
               time_segment='daymonth')

# Transacciones fraudulentas
fraude_txn = df.query('fraude == 1').copy()
circular_graph(fraude_txn, 
               title='Distribución transacciones fraudulentas por dia del mes',
               time_segment='daymonth')


fraude_txn.plot(x='hora', y='')

"""
En promedio las transacciones fraudulentas ocurren a las 3 p.m
los dias jueves
"""
#============================================
# Analisis Bivariado
#============================================

# Grafico de pares
sns.pairplot(data=num_df[['monto', 'hora', 'interes_tc', 
                          'dcto', 'cashback', 'dia_semana', 
                          'dia_mes', 'linea_tc', 'is_prime', 
                          'monto_neto', 'pct_cupo', 'fraude']],
             corner=True
            #  hue='fraude'
             )

# Coordenadas paralelas
cols = ['id_user', 'device_score', 'is_prime',
        'dia_semana', 'hora', 
        # 'linea_tc', 'monto', 'interes_tc', 
        'dcto', 'cashback',  'fraude']
plt.figure(figsize=(15, 10))
fig = px.parallel_coordinates(df, 
                              color="fraude", 
                              dimensions=cols,
                              #color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=0.5
                              )

fig.show()


# Categorias paralelas
cols = ['genero', 'tipo_tc', 'device_score', 'parte_dia', 'fraude']
fig = px.parallel_categories(cat_df[cols], 
                             color='fraude',
                             color_continuous_scale=px.colors.diverging.Tealrose)
fig.show()

"""
Las transacciones que son fraude tienden a ser de usuarios que no pagan servicios
adicionales (usuarios no primen), cashback hasta 10 unidades monetarias, con
descuentos hasta 100 unidades monetarias y con horas de transacciones entre
las 15 y 20. 

Tienden a realizar las transacciones con tarjeta de credito fisica
y tienden a ser más de genero hombre
"""

#============================================
# Seleccion de variable
#============================================
"""
Variables Potenciales
----------------------
- is_prime
- hora
- dcto
- cashback
- genero
- tipo_tc


Teniendo en cuenta el analisis univariado y bivariado realizado, se decide
eliminar para la creación del modelo las siguientes variables:

- device_os: tiene un porcentaje elevado de datos faltantes
- establecimiento: tiene un porcentaje elevado de datos faltantes
- ciudad: tiene un porcentaje elevado de datos faltantes
- dcto, cashback, monto: se crea la variable monto_neto que las embebe
- linea_tc: se crea la variable pct_cup que la embebe
- device_model: no tiene varianza, corresponde a un solo valor
- fecha, hora: se combinan para crear la variable fecha_hora
"""
df1 = df.drop(['device_os', 'device_score', 'establecimiento', 'ciudad', 
               'dcto', 'linea_tc', 'device_model',
               'cashback', 'monto', 'fecha', #'hora'
               ], axis=1)

df1.columns
cols = ['id_user', 'is_prime', 'hora', 
        'interes_tc', 'monto_neto', 'pct_cupo', 'fraude']
fig = px.parallel_coordinates(df1, 
                              color="fraude", 
                              dimensions=cols,
                              #color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=0.5
                              )

fig.show()

#============================================
# PCA
#============================================
user = num_df.id_user
target_var = num_df.fraude

num_df = num_df[['monto', 'linea_tc', 'interes_tc', 
                 'is_prime', 'dcto','cashback']]

# Escalamiento variables
scaler = StandardScaler()
scaled_num = scaler.fit_transform(num_df)
scaled_num = pd.DataFrame(scaled_num, columns=num_df.columns)


cat_df = cat_df[['genero', 'device_score', 'tipo_tc', 'status_txn']]
dummi_cat = pd.get_dummies(cat_df, drop_first=True)
pca_df = pd.concat([scaled_num, dummi_cat], axis=1)


pca = PCA(random_state=2022)
pca.fit(pca_df)
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

plt.plot(var_exp, marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.xticks(range(0, 14))
plt.show()

# Ciclo para identificar la cantidad de componentes que recogen el 80% de la varianza
for comp in range(2, pca_df.shape[1]):
    pca = PCA(n_components=comp, random_state=2023)
    pca.fit(pca_df) # Incluyendo categóricas
    variance_acum = pca.explained_variance_ratio_
    final_comp = comp
    
    threshold = 0.8 # varianza recogida deseada
    if round(variance_acum.sum(), 2) >= threshold:
        print(f"Cantidad de componentes que recoje el {threshold * 100}% de la varianza: {final_comp}")
        break

pca = PCA(n_components=5, random_state=2023)
pca.fit(pca_df)
data_pca = pca.transform(pca_df)
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

biplot(data_pca, pca.components_, 0, 1, pca_df.columns.tolist())
biplot(data_pca, pca.components_, 0, 2, pca_df.columns.tolist())
biplot(data_pca, pca.components_, 0, 3, pca_df.columns.tolist())
biplot(data_pca, pca.components_, 0, 4, pca_df.columns.tolist())

# Generar archivo procesado
df.to_csv(r'../../data/processed/transactions.csv',
          sep=',',
          encoding='utf-8',
          index=False)



df.loc[0, :].to_json()



