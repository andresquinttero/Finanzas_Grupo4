##### Importar librerias #####
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 as sql

############################ 1. LIMPIEZA Y TRANSFORMACION DE LOS DATOS ############################

# Conectarse a la base de datos 
conn = sql.connect('finanzas.db')

# Crear el cursor
cur = conn.cursor()

##### Cargar los archivos de datos #####
df_utilizaciones_medicas = pd.read_csv('databases/BD_UtilizacionesMedicas.csv', sep=';')
df_reclamaciones = pd.read_csv('databases/BD_Reclamaciones.csv', sep=';')
df_sociodemograficas = pd.read_csv('databases/BD_SocioDemograficas.csv', sep=';')
df_regional = pd.read_csv('databases/BD_Regional.csv', sep=';')
df_genero = pd.read_csv('databases/BD_Genero.csv', sep=';')
df_asegurados_expuestos = pd.read_csv('databases/BD_Asegurados_Expuestos.csv', sep=';')
df_diagnosticos = pd.read_csv('databases/BD_Diagnostico.csv', sep=';')


##### Explorar los datos iniciales para ver que se debe limpiar #####
dataframes = [df_utilizaciones_medicas, df_diagnosticos, df_reclamaciones, df_sociodemograficas, df_regional, df_genero, df_asegurados_expuestos]
nombres = ["Utilizaciones Medicas", "Diagnosticos", "Reclamaciones", "Socio Demograficas", "Regional", "Genero", "Asegurados Expuestos"]

# Imprimir los datos iniciales con un ciclo for para no repetir codigo
for df, nombre in zip(dataframes, nombres):
    print(f"{nombre}:")
    print("Primeras filas:")
    print(df.head())
    print("Forma (Filas, Variables):", df.shape)
    print("Valores nulos totales:", df.isnull().sum().sum())
    print("Datos duplicados:", df.duplicated().sum())
    print("-" * 80)

# Tras este primer vistazo se pueden ver que no hay datos nulos ni duplicados casi en nignuna BD
# Solo en la BD Asegurados Expuestos hay 126712 valores nulos totales, se procede a revisarlos
print("Asegurados Expuestos = Valores nulos por columna\n", df_asegurados_expuestos.isnull().sum())

# Se puede ver que la columna 'FECHA_CANCELACION' tiene 126712 valores nulos
# Se observa la columna mas detalladamente
print(df_asegurados_expuestos['FECHA_CANCELACION'].head(20))
# Vemos que muchos campos estan vacios, loq que puede indicar que muchas polizas no han sido canceladas hasta la fecha de corte de los datos.


##### Antes de decidir que hacer con estos datos, se procede a revisar los tipos de datos de cada BD #####

# Aplicar .info() a cada DataFrame
for df, nombre in zip(dataframes, nombres):
    print(f"{nombre}:")
    df.info()
    print("-" * 80)

# Aplicar .describe() a cada DataFrame
for df, nombre in zip(dataframes, nombres):
    print(f"{nombre}:")
    print(df.describe())
    print("-" * 80)


##### Se decide primero cambiar el tipo de dato a las columnas de fecha que lo requieran #####
# Convertir columnas de fecha en 'df_utilizaciones_medicas'
df_utilizaciones_medicas['Fecha_Reclamacion'] = pd.to_datetime(df_utilizaciones_medicas['Fecha_Reclamacion'], unit='D', origin='1899-12-30')

# Convertir columna de fecha en 'df_sociodemograficas'
df_sociodemograficas['FechaNacimiento'] = pd.to_datetime(df_sociodemograficas['FechaNacimiento'], unit='D', origin='1899-12-30')

# Convertir columnas de fecha en 'df_asegurados_expuestos'
df_asegurados_expuestos['FECHA_INICIO'] = pd.to_datetime(df_asegurados_expuestos['FECHA_INICIO'], unit='D', origin='1899-12-30')
df_asegurados_expuestos['FECHA_CANCELACION'] = pd.to_datetime(df_asegurados_expuestos['FECHA_CANCELACION'], unit='D', origin='1899-12-30', errors='coerce') # 'errors='coerce'' convertirá los valores no válidos (NaN) en NaT
df_asegurados_expuestos['FECHA_FIN'] = pd.to_datetime(df_asegurados_expuestos['FECHA_FIN'], unit='D', origin='1899-12-30')

# Mostrar las primeras filas de los dataframes para verificar los cambios
print(df_utilizaciones_medicas.head())
print(df_sociodemograficas.head())
print(df_asegurados_expuestos.head(50))
# Vemos que ya estan en el formato correcto


##### Ahora seguimos con el analisis de los datos nulos de la BD Asegurados Expuestos #####

# Fecha de corte (última fecha conocida en los datos)
fecha_corte = datetime(2019,12,31)

# Calcular la duración para pólizas no canceladas
df_asegurados_expuestos['Duracion'] = df_asegurados_expuestos.apply(lambda row: (fecha_corte - row['FECHA_INICIO']).days if pd.isna(row['FECHA_CANCELACION']) else (row['FECHA_CANCELACION'] - row['FECHA_INICIO']).days, axis=1)

# Mostrar las primeras filas para verificar los cambios
print(df_asegurados_expuestos.head())
# Vemos que las FECHA_CANCELACIOn nula duran lo mismo como si terminara en la FECHA_FIN, por lo que se procede a reemplazar los valores nulos por la FECHA_FIN
df_asegurados_expuestos['FECHA_CANCELACION'] = df_asegurados_expuestos['FECHA_CANCELACION'].fillna(df_asegurados_expuestos['FECHA_FIN'])

# Mostrar las primeras filas para verificar los cambios
print(df_asegurados_expuestos.head(50))
# Y vemos que ya no hay valores nulos en la columna FECHA_CANCELACION
df_asegurados_expuestos.isnull().sum()


##### Continuaremos con la limpiza y transformacion de los datos #####

# Unir BD segun sea necesario
df_merged = pd.merge(df_utilizaciones_medicas, df_sociodemograficas, on='Afiliado_Id', how='left')
df_merged = pd.merge(df_merged, df_diagnosticos, on='Diagnostico_Codigo', how='left')
df_merged = pd.merge(df_merged, df_reclamaciones, on='Reclamacion_Cd', how='left')
df_merged = pd.merge(df_merged, df_regional, left_on='Regional', right_on='Regional_Id', how='left')
df_merged = pd.merge(df_merged, df_genero, left_on='Sexo_Cd', right_on='Sexo_Cd', how='left')
df_merged = pd.merge(df_merged, df_asegurados_expuestos, left_on='Afiliado_Id', right_on='Asegurado_Id', how='left')

print(df_merged.head())

# Se procede a Calcular la edad de los afiliados en años segundos la fecha de nacimiento y la fecha de corte y se anade al dataframe df_merged
df_merged['Edad'] = df_merged.apply(lambda row: (fecha_corte - row['FechaNacimiento']).days // 365, axis=1)

##### Luego de observar las diferenes BD se puede analizar lo siguiente: #####

# 1. BD Utilizaciones Medicas:
# - La distribución de Numero_Utilizaciones y Valor_Utilizaciones sugiere una gran variabilidad en los costos y la frecuencia de las utilizaciones médicas. 
# Es relevante investigar más sobre los casos con altos números de utilizaciones o valores extremadamente altos, ya que pueden ser outliers o casos especiales.

# 2. BD Diagnosticos:   
# Se pueden explorar la relación entre los códigos de diagnóstico y las utilizaciones médicas para entender mejor las condiciones de salud más costosas o frecuentes.

# 3. BD Reclamaciones:
# La tabla es pequeña y manejable. Se podrian relacionar los códigos de reclamación con los datos de utilizaciones médicas para analizar los tipos de reclamaciones más comunes o costosas.

# 4. BD Socio Demograficas:
# La presencia de enfermedades como cáncer, EPOC, diabetes, hipertensión y enfermedades cardiovasculares podría correlacionarse con un mayor uso de servicios médicos.
# Sería útil investigar cómo estas condiciones afectan las reclamaciones y los costos.

# 5. BD Regional Y Genero:
# Estos datos pueden ser útiles para análisis demográficos y regionales, para ver si hay diferencias en las utilizaciones y costos de servicios médicos.

# 6. BD Asegurados Expuestos:
# Se podria investigar cómo la duración de las pólizas (calculada a partir de las fechas de inicio y cancelación) se relaciona con las reclamaciones y utilizaciones médicas.

# 7. Analisis de datos adicionales:
# Realizar análisis estadísticos para identificar correlaciones entre variables, como edad, género, regional, y las utilizaciones médicas y reclamaciones.
# Considerar la integración de los diferentes conjuntos de datos para un análisis más holístico, lo que podría requerir transformaciones, como la unión de tablas basadas en claves comunes (por ejemplo, Afiliado_Id).



############################ 2. EXPLORACION DE LOS DATOS ############################

### a. Distribución de Utilizaciones y Costos ###
sns.boxplot(x=df_merged['Numero_Utilizaciones'])
plt.show()
# Aqui se muestra la distribución de la cantidad de servicios médicos utilizados por los afiliados.
# Se observa que La mayoría de las utilizaciones se concentran en un rango bajo.
# Existen algunos puntos que son considerados outliers, situados lejos de la mayoría de los datos. Estos representan afiliados que han tenido un número de utilizaciones mucho mayor que la media.

# Se priocede a hacer un analisis estadistico de los datos de la columna 'Numero_Utilizaciones'
print(df_merged['Numero_Utilizaciones'].describe())
# La cantidad de servicios médicos utilizados esta con una media de 1.57, lo que indica que la mayoría de los afiliados tienen un número bajo de utilizaciones.
# Ademas que el 75% de los afiliados tienen 1 o menos utilizaciones.

#Ahora vemos mas a profundidad los datos de los outliers, y hacemos la cuenta de cuantos tienen mas de 10 utilizaciones
print(df_merged[df_merged['Numero_Utilizaciones'] > 10].head(10))
print(df_merged[df_merged['Numero_Utilizaciones'] > 10].shape)
# Vemos que 18299 afiliados tienen mas de 10 utilizaciones, lo que representa el 2,21% de los afiliados.
# No es un valor muy significativo pero se analizaran de todas formas. Sobretodo los mas extremos.

outliers_extremos = df_merged[df_merged['Numero_Utilizaciones'] > 50]
print("Cantidad de outliers extremos:", outliers_extremos.shape[0])
print(outliers_extremos.head(14))

# 1. De esto podemos conlcluir que los afiliados con un alto número de utilizaciones tienen edades que varían desde niños hasta personas de edad avanzada. 
# Por ejemplo, hay registros de afiliados que solo tienen 7 años, y otros que tienen 96 años. Esto podría sugerir que no hay un grupo de edad específico asociado con un alto uso de servicios médicos, sino que podría depender de condiciones individuales de salud 

# 2. Algunos afiliados aparecen más de una vez, como el afiliado con ID 55030322. Esto podría indicar que están recibiendo servicios médicos repetitivos o que hay duplicados en los datos que necesitan ser revisados. Y con varios valores nulos. Se procede a elimiar el afiliado con ID 44269959 debido a sus valores nulos.
df_merged = df_merged[df_merged['Afiliado_Id'] != 44269959]
outliers_extremos = df_merged[df_merged['Numero_Utilizaciones'] > 50]

# 3. Los outliers extremos no están concentrados en una sola región, lo que puede descartar la posibilidad de un error sistemático asociado con una ubicación en particular

# 4. Hay algunas inconsistencias, como registros de personas que tienen 210 utilizaciones pero un valor de utilizaciones relativamente bajo. Esto podría sugerir errores de entrada o situaciones particulares donde se registraron muchas utilizaciones individuales con bajo costo cada una.

# Para obtener una mejor comprensión de estos casos extremos, se procede a analizar y graficar más variables relacionadas.


# Gráfico de dispersión de Numero_Utilizaciones vs Edad
sns.scatterplot(data=outliers_extremos, x='Edad', y='Numero_Utilizaciones')
plt.title('Relación entre Edad y Número de Utilizaciones para Outliers Extremos')
plt.xlabel('Edad')
plt.ylabel('Número de Utilizaciones')
plt.show()
# Aqui vemos que los 2 valores mas altos estan cercanos a los 60 años, y que la mayoria de los outliers extremos estan entre los 0 y 20 años.


# Gráfico de dispersión de Numero_Utilizaciones vs Valor_Utilizaciones
sns.scatterplot(data=outliers_extremos, x='Numero_Utilizaciones', y='Valor_Utilizaciones')
plt.title('Relación entre Número y Valor de Utilizaciones para Outliers Extremos')
plt.xlabel('Número de Utilizaciones')
plt.ylabel('Valor de Utilizaciones')
plt.show()
# La relación entre el número de utilizaciones y su valor no parece ser lineal ni directamente proporcional. 
# Esto puede sugerir que hay casos en los que muchas utilizaciones no necesariamente se traducen en un costo más alto. Esto podría deberse a que algunas utilizaciones pueden ser servicios de menor costo.

# Visualización de la distribución de outliers por región
plt.figure(figsize=(10, 6))
sns.countplot(data=outliers_extremos, x='Regional_desc')
plt.title('Distribución de Outliers Extremos por Región')
plt.xticks(rotation=45)
plt.xlabel('Región')
plt.ylabel('Cantidad de Outliers')
plt.show()

# Distribución de los valores de utilización entre los outliers extremos.
plt.figure(figsize=(10, 6))
sns.histplot(outliers_extremos['Valor_Utilizaciones'], bins=30, kde=True)
plt.title('Distribución del Valor de Utilizaciones entre Outliers Extremos')
plt.xlabel('Valor de Utilizaciones')
plt.ylabel('Frecuencia')
plt.show()


######## Ahora se procede a hacer el mismo analisis pero para TODOS los afiliados ########
# Gráfico de dispersión de Numero_Utilizaciones vs Edad para todos los datos
sns.scatterplot(data=df_merged, x='Edad', y='Numero_Utilizaciones')
plt.title('Relación entre Edad y Número de Utilizaciones')
plt.xlabel('Edad')
plt.ylabel('Número de Utilizaciones')
plt.show()

# Gráfico de dispersión de Numero_Utilizaciones vs Valor_Utilizaciones para todos los datos
sns.scatterplot(data=df_merged, x='Numero_Utilizaciones', y='Valor_Utilizaciones')
plt.title('Relación entre Número y Valor de Utilizaciones')
plt.xlabel('Número de Utilizaciones')
plt.ylabel('Valor de Utilizaciones')
plt.show()

# Visualización de la distribución de outliers por región
plt.figure(figsize=(10, 6))
sns.countplot(data=df_merged, x='Regional_desc')
plt.title('Distribución por Región')
plt.xticks(rotation=45)
plt.xlabel('Región')
plt.ylabel('Cantidad de Pacientes')
plt.show()

# Guardar df_merged en la base de datos SQL para luego usarla en otros archivos
df_merged.to_sql('df_merged', conn, index=False, if_exists='replace')

# Cerrar la conexión a la base de datos
conn.close()
