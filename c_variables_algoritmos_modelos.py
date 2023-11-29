##### Importar librerias #####
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 as sql
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression    
from sklearn.tree import DecisionTreeRegressor       
from sklearn.ensemble import RandomForestRegressor    
from sklearn.linear_model import SGDRegressor       
from sklearn.model_selection import KFold, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import joblib


# Conectarse a la base de datos 
conn = sql.connect('finanzas.db')

# Crear el cursor
cur = conn.cursor()

# Para ver las tablas que hay en la base de datos
cur.execute("SELECT name FROM sqlite_master where type='table' ")
cur.fetchall() 

# Cargamos los datos del dataframe df_merged
df_merged = pd.read_sql_query("SELECT * FROM df_merged", conn)
df_merged.head()

# Grafico de las edades que más reclamaciones realizan

df_edad = df_merged.groupby(['Edad'])['Numero_Utilizaciones'].sum().reset_index().sort_values(by = 'Numero_Utilizaciones',ascending = False)
fig = px.bar(df_edad, x='Edad', y="Numero_Utilizaciones", color='Numero_Utilizaciones', title="<b>Edad de quienes más reclaman<b>", barmode = 'group')
fig.update_layout(
    xaxis_title = 'Edad',
    yaxis_title = 'N° de Reclamaciones',
    template = 'simple_white',
    title_x = 0.5)
fig.show()

# Las edades con más reclamaciones están entre los 29 y 44 años, este rango será el enfoque de análisis para los modelos
df_merged.groupby(['Edad'])[['Numero_Utilizaciones']].sum().reset_index().query("Edad > 28 & Edad < 45").reset_index(drop = True).sort_values(by = 'Edad',ascending = True)
df_modelos = df_merged.query("Edad >= 29 & Edad <= 44").reset_index(drop = True)
df_modelos.head()
df_modelos.shape

df_modelos.isnull().sum()

# Selección de variables para los modelos
df_models = df_modelos[['Valor_Utilizaciones', 'Reclamacion_Desc', 'Edad', 'Sexo_Cd',
                        'CANCER', 'EPOC', 'DIABETES', 'HIPERTENSION', 'ENF_CARDIOVASCULAR',
                        'Regional_desc']]

# Pasamos a dummies las categóricas
reclamacion_dummies = pd.get_dummies(df_models['Reclamacion_Desc'], prefix = 'Reclamacion')
sexo_dummies = pd.get_dummies(df_models['Sexo_Cd'], prefix = 'Sexo')
regional_dummies = pd.get_dummies(df_models['Regional_desc'], prefix = 'Regional')

# Unimos las dummies con el df con el que vamos a trabajar
df_models = pd.concat([df_models, reclamacion_dummies, sexo_dummies, regional_dummies], axis = 1)

# Eliminamos las columnas que ya no vamos a utilizar
df_models = df_models.drop(['Reclamacion_Desc', 'Sexo_Cd', 'Regional_desc'], axis = 1)

###### Escalamos las variables #####

# Para las columnas de Valor_Utilizaciones y Edad
scale_transform = ColumnTransformer([('escaladas', MinMaxScaler(feature_range=(0,1)), df_models.columns)])
scale_transform

# Se escalan las columnas
models_escalado = scale_transform.fit_transform(df_models)

# Se crea el dataframe con las columnas escaladas
df_models_escalado = pd.DataFrame(models_escalado, columns = df_models.columns)

###### Selección de características ######

df_models_escalado.shape

# Arreglo del modelo
arreglo = df_models_escalado.values
X = arreglo[:,1:]
Y = arreglo[:,0]

# Selección de características de acuerdo con las k best

test = SelectKBest(score_func = f_classif, k = 10) # Se seleccionan las 10 mejores características
fit = test.fit(X, Y)
print("Puntaje KBest \n",fit.scores_) # Se muestran los resultados

mask = fit.get_support() # Se crea una máscara con las características seleccionadas
nombres_columnas = df_models_escalado.columns # Lista con los nombres de las columnas

nombres_caracteristicas = [nombre for nombre, valor in zip(nombres_columnas, mask) if valor] # Se crea una lista con las características seleccionadas emparejando los nombres de las columnas con los valores de la máscara
nombres_caracteristicas

df_features = df_models_escalado.loc[:,nombres_caracteristicas] # df con las características seleccionadas

##### Selección del modelo #####

arreglo = df_models_escalado.values
X = arreglo[:,1:]
y = arreglo[:,:1]

# Evaluación de modelos con la métrica de precisión MSE
listaModelos = []
listaModelos.append(('LR',LinearRegression())) 
listaModelos.append(('DT',DecisionTreeRegressor()))
listaModelos.append(('RFR',RandomForestRegressor()))
listaModelos.append(('SGDR',SGDRegressor()))

listaResultados = []
nombres = []
scoring = 'neg_mean_squared_error'

for nom, modelo in listaModelos:
  kfold = KFold(n_splits=5, random_state=42, shuffle=True)
  res_cv = cross_val_score(modelo, X, Y, cv=kfold, scoring=scoring)
  listaResultados.append(res_cv)
  nombres.append(nom)
  print(nom, res_cv.mean()*-1, res_cv.std())
print(nombres)
print(listaResultados)
print(res_cv.mean())

##### Ajuste de hiperparámetros ######


param_grid = {'bootstrap': [True, False],
    'max_depth': [10, 60],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5],
    'n_estimators': [3, 5]
}

hyp_RFR = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_grid, n_iter=20, scoring="neg_mean_squared_error")
hyp_RFR.fit(X, Y)

resultados = hyp_RFR.cv_results_
best_params = hyp_RFR.best_params_
pd_resultados = pd.DataFrame(resultados)
best_score = pd_resultados['mean_test_score'].max()

print("Best parameters:", best_params)
print("Best score:", best_score)

modelo_RFR =  hyp_RFR.best_estimator_         ### Modelo con los mejores hiperparametros
modelo_RFR

##### Evaluación del modelo #####

# Modelo con hiperparámetros
eva_mod1 = cross_validate(modelo_RFR,X,Y,cv = 5, scoring ="neg_mean_squared_error", return_train_score = True)
eva_mod1 = pd.DataFrame(eva_mod1)
eva_mod1.mean()

# Modelo sin hiperparámetros
eva_mod2 = cross_validate(RandomForestRegressor(),X,Y,cv = 5, scoring ="neg_mean_squared_error", return_train_score = True)
eva_mod2 = pd.DataFrame(eva_mod2)
eva_mod2.mean()


#### Despliegue del modelo #####

# Guardar el modelo
joblib.dump(modelo_RFR, 'modelo_RFR.pkl')

# Cargar el modelo en producción
modelo_cargado = joblib.load('modelo_RFR.pkl')
