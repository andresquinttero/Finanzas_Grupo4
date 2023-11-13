##### Importar librerias #####
import pandas as pd
import numpy as np
from datetime import datetime


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


##### Luego de observar las diferenes BD se puede analizar lo siguiente: #####

# 1. BD Utilizaciones Medicas:
# - La distribución de Numero_Utilizaciones y Valor_Utilizaciones sugiere una gran variabilidad en los costos y la frecuencia de las utilizaciones médicas. 
# Es relevante investigar más sobre los casos con altos números de utilizaciones o valores extremadamente altos, ya que pueden ser outliers o casos especiales.
# - En la  la columna 'FECHA' es de tipo object, se debe convertir a tipo fecha

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



##### Analisis de Outliers ##### AUN NO HECHO
# Cálculo de IQR para Numero_Utilizaciones
Q1_num = df_utilizaciones_medicas['Numero_Utilizaciones'].quantile(0.25)
Q3_num = df_utilizaciones_medicas['Numero_Utilizaciones'].quantile(0.75)
IQR_num = Q3_num - Q1_num

# Filtrar outliers
outliers_num = df_utilizaciones_medicas[(df_utilizaciones_medicas['Numero_Utilizaciones'] < (Q1_num - 1.5 * IQR_num)) | 
                                       (df_utilizaciones_medicas['Numero_Utilizaciones'] > (Q3_num + 1.5 * IQR_num))]

# Repetir el proceso para Valor_Utilizaciones
Q1_val = df_utilizaciones_medicas['Valor_Utilizaciones'].quantile(0.25)
Q3_val = df_utilizaciones_medicas['Valor_Utilizaciones'].quantile(0.75)
IQR_val = Q3_val - Q1_val

outliers_val = df_utilizaciones_medicas[(df_utilizaciones_medicas['Valor_Utilizaciones'] < (Q1_val - 1.5 * IQR_val)) | 
                                       (df_utilizaciones_medicas['Valor_Utilizaciones'] > (Q3_val + 1.5 * IQR_val))]

# Ver los outliers
print(outliers_num)
print(outliers_val)
