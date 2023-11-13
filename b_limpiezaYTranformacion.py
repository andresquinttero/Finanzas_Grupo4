# Importar librerias
import pandas as pd


# Cargar los archivos de datos
df_utilizaciones_medicas = pd.read_csv('databases/BD_UtilizacionesMedicas.csv')
df_reclamaciones = pd.read_csv('databases/BD_Reclamaciones.csv')
df_sociodemograficas = pd.read_csv('databases/BD_SocioDemograficas.csv')
df_regional = pd.read_csv('databases/BD_Regional.csv')
df_genero = pd.read_csv('databases/BD_Genero.csv')
df_asegurados_expuestos = pd.read_csv('databases/BD_Asegurados_Expuestos.csv')
# Esta base de datos nos esta dando error al cargarla por su separador, se procede a corregirlo
df_diagnosticos = pd.read_csv('databases/BD_Diagnostico.csv', sep=';')


# Explorar los datos iniciales para ver que se debe limpiar
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

# Tras este primer vistazo se pueden ver que no hay datos nulos ni duplicados en ninguna de las bases de datos
