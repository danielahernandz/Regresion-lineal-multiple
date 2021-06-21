#!/usr/bin/env python
# coding: utf-8

# In[14]:


#https://www.cienciadedatos.net/documentos/py10-regresion-lineal-python.html
#Regresión lineal múltiple
"""Supóngase que el departamento de ventas de una empresa quiere estudiar la influencia que tiene la publicidad
a través de distintos canales sobre el número de ventas de un producto. Se dispone de un conjunto de datos que 
contiene los ingresos (en millones) conseguido por ventas en 200 regiones, así como la cantidad de presupuesto, 
también en millones, destinado a anuncios por radio, TV y periódicos en cada una de ellas."""


# In[15]:


#Librerías
#Tratamiento de datos 
#========================================================================================
import pandas as pd 
import numpy as np

#Graficos 
#========================================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns 

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats


# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


# In[16]:


# Datos
# ==============================================================================
tv = [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8, 66.1, 214.7,
      23.8, 97.5, 204.1, 195.4, 67.8, 281.4, 69.2, 147.3, 218.4, 237.4, 13.2,
      228.3, 62.3, 262.9, 142.9, 240.1, 248.8, 70.6, 292.9, 112.9, 97.2, 265.6,
      95.7, 290.7, 266.9, 74.7, 43.1, 228.0, 202.5, 177.0, 293.6, 206.9, 25.1,
      175.1, 89.7, 239.9, 227.2, 66.9, 199.8, 100.4, 216.4, 182.6, 262.7, 198.9,
      7.3, 136.2, 210.8, 210.7, 53.5, 261.3, 239.3, 102.7, 131.1, 69.0, 31.5,
      139.3, 237.4, 216.8, 199.1, 109.8, 26.8, 129.4, 213.4, 16.9, 27.5, 120.5,
      5.4, 116.0, 76.4, 239.8, 75.3, 68.4, 213.5, 193.2, 76.3, 110.7, 88.3, 109.8,
      134.3, 28.6, 217.7, 250.9, 107.4, 163.3, 197.6, 184.9, 289.7, 135.2, 222.4,
      296.4, 280.2, 187.9, 238.2, 137.9, 25.0, 90.4, 13.1, 255.4, 225.8, 241.7, 175.7,
      209.6, 78.2, 75.1, 139.2, 76.4, 125.7, 19.4, 141.3, 18.8, 224.0, 123.1, 229.5,
      87.2, 7.8, 80.2, 220.3, 59.6, 0.7, 265.2, 8.4, 219.8, 36.9, 48.3, 25.6, 273.7,
      43.0, 184.9, 73.4, 193.7, 220.5, 104.6, 96.2, 140.3, 240.1, 243.2, 38.0, 44.7,
      280.7, 121.0, 197.6, 171.3, 187.8, 4.1, 93.9, 149.8, 11.7, 131.7, 172.5, 85.7,
      188.4, 163.5, 117.2, 234.5, 17.9, 206.8, 215.4, 284.3, 50.0, 164.5, 19.6, 168.4,
      222.4, 276.9, 248.4, 170.2, 276.7, 165.6, 156.6, 218.5, 56.2, 287.6, 253.8, 205.0,
      139.5, 191.1, 286.0, 18.7, 39.5, 75.5, 17.2, 166.8, 149.7, 38.2, 94.2, 177.0,
      283.6, 232.1]

radio = [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6, 5.8, 24.0, 35.1,
         7.6, 32.9, 47.7, 36.6, 39.6, 20.5, 23.9, 27.7, 5.1, 15.9, 16.9, 12.6, 3.5,
         29.3, 16.7, 27.1, 16.0, 28.3, 17.4, 1.5, 20.0, 1.4, 4.1, 43.8, 49.4, 26.7,
         37.7, 22.3, 33.4, 27.7, 8.4, 25.7, 22.5, 9.9, 41.5, 15.8, 11.7, 3.1, 9.6,
         41.7, 46.2, 28.8, 49.4, 28.1, 19.2, 49.6, 29.5, 2.0, 42.7, 15.5, 29.6, 42.8,
         9.3, 24.6, 14.5, 27.5, 43.9, 30.6, 14.3, 33.0, 5.7, 24.6, 43.7, 1.6, 28.5,
         29.9, 7.7, 26.7, 4.1, 20.3, 44.5, 43.0, 18.4, 27.5, 40.6, 25.5, 47.8, 4.9,
         1.5, 33.5, 36.5, 14.0, 31.6, 3.5, 21.0, 42.3, 41.7, 4.3, 36.3, 10.1, 17.2,
         34.3, 46.4, 11.0, 0.3, 0.4, 26.9, 8.2, 38.0, 15.4, 20.6, 46.8, 35.0, 14.3,
         0.8, 36.9, 16.0, 26.8, 21.7, 2.4, 34.6, 32.3, 11.8, 38.9, 0.0, 49.0, 12.0,
         39.6, 2.9, 27.2, 33.5, 38.6, 47.0, 39.0, 28.9, 25.9, 43.9, 17.0, 35.4, 33.2,
         5.7, 14.8, 1.9, 7.3, 49.0, 40.3, 25.8, 13.9, 8.4, 23.3, 39.7, 21.1, 11.6, 43.5,
         1.3, 36.9, 18.4, 18.1, 35.8, 18.1, 36.8, 14.7, 3.4, 37.6, 5.2, 23.6, 10.6, 11.6,
         20.9, 20.1, 7.1, 3.4, 48.9, 30.2, 7.8, 2.3, 10.0, 2.6, 5.4, 5.7, 43.0, 21.3, 45.1,
         2.1, 28.7, 13.9, 12.1, 41.1, 10.8, 4.1, 42.0, 35.6, 3.7, 4.9, 9.3, 42.0, 8.6]

periodico = [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 1.0, 21.2, 24.2, 4.0,
             65.9, 7.2, 46.0, 52.9, 114.0, 55.8, 18.3, 19.1, 53.4, 23.5, 49.6, 26.2,
             18.3, 19.5, 12.6, 22.9, 22.9, 40.8, 43.2, 38.6, 30.0, 0.3, 7.4, 8.5, 5.0,
             45.7, 35.1, 32.0, 31.6, 38.7, 1.8, 26.4, 43.3, 31.5, 35.7, 18.5, 49.9,
             36.8, 34.6, 3.6, 39.6, 58.7, 15.9, 60.0, 41.4, 16.6, 37.7, 9.3, 21.4, 54.7,
             27.3, 8.4, 28.9, 0.9, 2.2, 10.2, 11.0, 27.2, 38.7, 31.7, 19.3, 31.3, 13.1,
             89.4, 20.7, 14.2, 9.4, 23.1, 22.3, 36.9, 32.5, 35.6, 33.8, 65.7, 16.0, 63.2,
             73.4, 51.4, 9.3, 33.0, 59.0, 72.3, 10.9, 52.9, 5.9, 22.0, 51.2, 45.9, 49.8,
             100.9, 21.4, 17.9, 5.3, 59.0, 29.7, 23.2, 25.6, 5.5, 56.5, 23.2, 2.4, 10.7,
             34.5, 52.7, 25.6, 14.8, 79.2, 22.3, 46.2, 50.4, 15.6, 12.4, 74.2, 25.9, 50.6,
             9.2, 3.2, 43.1, 8.7, 43.0, 2.1, 45.1, 65.6, 8.5, 9.3, 59.7, 20.5, 1.7, 12.9,
             75.6, 37.9, 34.4, 38.9, 9.0, 8.7, 44.3, 11.9, 20.6, 37.0, 48.7, 14.2, 37.7,
             9.5, 5.7, 50.5, 24.3, 45.2, 34.6, 30.7, 49.3, 25.6, 7.4, 5.4, 84.8, 21.6, 19.4,
             57.6, 6.4, 18.4, 47.4, 17.0, 12.8, 13.1, 41.8, 20.3, 35.2, 23.7, 17.6, 8.3,
             27.4, 29.7, 71.8, 30.0, 19.6, 26.6, 18.2, 3.7, 23.4, 5.8, 6.0, 31.6, 3.6, 6.0,
             13.8, 8.1, 6.4, 66.2, 8.7]

ventas = [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6, 8.6, 17.4, 9.2, 9.7,
          19.0, 22.4, 12.5, 24.4, 11.3, 14.6, 18.0, 12.5, 5.6, 15.5, 9.7, 12.0, 15.0, 15.9,
          18.9, 10.5, 21.4, 11.9, 9.6, 17.4, 9.5, 12.8, 25.4, 14.7, 10.1, 21.5, 16.6, 17.1,
          20.7, 12.9, 8.5, 14.9, 10.6, 23.2, 14.8, 9.7, 11.4, 10.7, 22.6, 21.2, 20.2, 23.7,
          5.5, 13.2, 23.8, 18.4, 8.1, 24.2, 15.7, 14.0, 18.0, 9.3, 9.5, 13.4, 18.9, 22.3,
          18.3, 12.4, 8.8, 11.0, 17.0, 8.7, 6.9, 14.2, 5.3, 11.0, 11.8, 12.3, 11.3, 13.6,
          21.7, 15.2, 12.0, 16.0, 12.9, 16.7, 11.2, 7.3, 19.4, 22.2, 11.5, 16.9, 11.7, 15.5,
          25.4, 17.2, 11.7, 23.8, 14.8, 14.7, 20.7, 19.2, 7.2, 8.7, 5.3, 19.8, 13.4, 21.8,
          14.1, 15.9, 14.6, 12.6, 12.2, 9.4, 15.9, 6.6, 15.5, 7.0, 11.6, 15.2, 19.7, 10.6,
          6.6, 8.8, 24.7, 9.7, 1.6, 12.7, 5.7, 19.6, 10.8, 11.6, 9.5, 20.8, 9.6, 20.7, 10.9,
          19.2, 20.1, 10.4, 11.4, 10.3, 13.2, 25.4, 10.9, 10.1, 16.1, 11.6, 16.6, 19.0, 15.6,
          3.2, 15.3, 10.1, 7.3, 12.9, 14.4, 13.3, 14.9, 18.0, 11.9, 11.9, 8.0, 12.2, 17.1,
          15.0, 8.4, 14.5, 7.6, 11.7, 11.5, 27.0, 20.2, 11.7, 11.8, 12.6, 10.5, 12.2, 8.7,
          26.2, 17.6, 22.6, 10.3, 17.3, 15.9, 6.7, 10.8, 9.9, 5.9, 19.6, 17.3, 7.6, 9.7, 12.8,
          25.5, 13.4]

datos = pd.DataFrame({'tv': tv, 'radio': radio, 'periodico':periodico, 'ventas': ventas})


# In[17]:


#Relación entre variables
"""El primer paso a la hora de establecer un modelo lineal múltiple es estudiar la relación que existe entre variables. 
Esta información es crítica a la hora de identificar cuáles pueden ser los mejores predictores para el modelo, 
y para detectar colinealidad entre predictores. A modo complementario, es recomendable representar la distribución 
de cada variable mediante histogramas."""


# In[18]:


# Correlación entre columnas numéricas
# ==============================================================================

def tidy_corr_matrix(corr_mat):
    ''' Función para convertir una matriz de correlación de pandas en formato tidy
    '''
#Los datos ordenados o 'tidy data' son los que se obtienen a partir de un proceso llamado 'data tidying' u ordenamiento de datos.
#1-Es uno de los procesos de limpieza importantes durante procesamiento de grandes datos o 'big data'.
#Los conjuntos de datos ordenados tienen una estructura que facilita el trabajo; son sencillos de manipular, modelar 
#y visualizar. Conjuntos de datos 'tidy' están ordenados de tal manera que cada variable es una columna y cada observación
#(o caso) es una fila.

    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

corr_matrix = datos.select_dtypes(include=['float64', 'int']).corr(method='pearson')
tidy_corr_matrix(corr_matrix).head(10)


# In[19]:


# Heatmap (mapa de color) matriz de correlaciones
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4)) #Forma matricial un cuadro de dimension 4x4 

#Mapa de calor seaborn 
sns.heatmap(
    corr_matrix,
    annot     = True,
    cbar      = False,
    annot_kws = {"size": 10}, #Tamaño numeros
    vmin      = -1, #Valor minimo de la matriz 
    vmax      = 1, #Valor max de la matriz
    center    = 0, #Intensidad de color de 1 a -- el mapa toma tono rosa
    cmap      = sns.diverging_palette(20, 300, n=100), #paleta de colores 300 es lila - 220 azul - 100 verde 
    square    = True, #Si es Verdadero, establezca el aspecto de los ejes en "igual" para que cada celda tenga forma cuadrada.
    ax        = ax
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right', # puse left y no hubo cambio total es matriz
)

ax.tick_params(labelsize = 12) #Tamaño letra 


# In[20]:


"""
seaborn.heatmap( data , * , vmin = None , vmax = None , cmap = None , center = None , robust = False , annot = None , fmt = '.2g' , annot_kws = None , linewidths = 0 , linecolor = 'white' , cbar = Verdadero , cbar_kws = Ninguno ,cbar_ax = Ninguno , cuadrado = Falso , xticklabels = 'auto' , yticklabels = 'auto' , máscara = Ninguno , ax = Ninguno , ** kwargs ) 
Trace datos rectangulares como una matriz codificada por colores.

Esta es una función de nivel de ejes y dibujará el mapa de calor en los ejes actualmente activos si no se proporciona
ninguno al ax argumento. Parte de este espacio de ejes se tomará y se utilizará para trazar un mapa de colores, 
a menos que cbar sea ​​FALSE o se proporcione un eje por separado cbar_ax.


-Parameters
datarectangular dataset
Conjunto de datos 2D que se puede convertir en un ndarray. 
Si se proporciona un DataFrame de Pandas, la información del índice / columna se utilizará para etiquetar las columnas y filas.

(Un ndarray es un contenedor multidimensional (generalmente de tamaño fijo) de elementos del mismo tipo y tamaño. 
El número de dimensiones y elementos de una matriz se define por su forma, que es una tupla de N números enteros no negativos que especifican los tamaños de cada dimensión.)

-vmin, vmax floats, optional
Valores para anclar el mapa de colores; de lo contrario, se infieren de los datos y otros argumentos de palabras clave.

-cmapmatplotlib colormap name or object, or list of colors, optional

El mapeo de los valores de los datos al espacio de color. 
Si no se proporciona, el valor predeterminado dependerá de si _center_  está configurado.

-centerfloat, optional
El valor en el que centrar el mapa de colores al trazar datos divergentes. 
El uso de este parámetro cambiará el valor predeterminado _cmap_ si no se especifica ninguno.

-robustbool, optional
Si True y / vmin o vmax están ausentes, 
el rango del mapa de colores se calcula con cuantiles robustos en lugar de los valores extremos.

-annotbool or rectangular dataset, optional
Si es verdadero, escriba el valor de los datos en cada celda. 
Si es una matriz con la misma forma que data, utilícela para anotar el mapa de calor en lugar de los datos. 
Tenga en cuenta que DataFrames coincidirá en la posición, no en el índice.

-fmt: str, optional
Código de formato de cadena para usar al agregar anotaciones.

-annot_kws: dict of key, value mappings, optional
Argumentos de palabras clave para matplotlib.axes.Axes.text()cuando annot es verdadero.

-linewidths: float, optional
Ancho de las líneas que dividirán cada celda.

-linecolor: color, optional
Color de las líneas que dividirán cada celda.

-cbar: bool, optional
Ya sea para dibujar una barra de colores.

-cbar_kws: dict of key, value mappings, optional
Argumentos de palabras clave para matplotlib.figure.Figure.colorbar().

-cbar_ax: matplotlib Axes, optional
Ejes en los que dibujar la barra de colores; de lo contrario, ocupa espacio de los ejes principales.

-squar: bool, optional
Si es Verdadero, establezca el aspecto de los ejes en "igual" para que cada celda tenga forma cuadrada.

-xticklabels, yticklabels: “auto”, bool, list-like, or int, optional
Si es Verdadero, grafica los nombres de las columnas del marco de datos. Si es falso, no grafique los nombres de las columnas. Si es similar a una lista, trace estas etiquetas alternativas como xticklabels. Si es un número entero, use los nombres de las columnas, pero grafique solo cada n etiqueta. 
Si es "automático", intente trazar densamente etiquetas que no se superpongan.

-mask: bool array or DataFrame, optional
Si se pasa, los datos no se mostrarán en las celdas donde _mask_ es Verdadero. 
Las celdas con valores perdidos se enmascaran automáticamente

-ax: matplotlib Axes, optional
Ejes en los que dibujar la trama; de lo contrario, utilice los ejes actualmente activos.

-kwargs: other keyword arguments
Ejes en los que dibujar la trama; de lo contrario, utilice los ejes actualmente activos.

Returns
ax: matplotlib Axes
Objeto de ejes con el mapa de calor.
https://seaborn.pydata.org/generated/seaborn.heatmap.html
"""


# In[21]:


# Gráfico de distribución para cada variable numérica
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 5))  #(ancho horizontal, ancho vertical) #nrows:2 - ncols:2 ==4 graficos
axes = axes.flat # no es una función, es un atributo de numpy. Al ser un interador sobre la matriz, puede usarlo para recorrer todos los ejes de la matriz de ejes
columnas_numeric = datos.select_dtypes(include=['float64', 'int']).columns

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data    = datos,
        x       = colum,
        stat    = "count",
        kde     = True,
        color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws= {'linewidth': 3}, #Tamaño linea tendencia
        alpha   = 0.3, #Transpariencia del color de las barras 
        ax      = axes[i]
    )
    axes[i].set_title(colum, fontsize = 10, fontweight = "bold") #tamaño letra color negrita "bold"
    axes[i].tick_params(labelsize = 10) #Tamaño numeros 
    axes[i].set_xlabel("")


    
fig.tight_layout()
plt.subplots_adjust(top = 0.9) #tamalo general del grafico
fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold");


# In[22]:


#Ajuste del modelo
#Se ajusta un modelo lineal múltiple con el objetivo de predecir las ventas en función 
#de la inversión en los tres canales de publicidad.


# In[23]:


# División de los datos en train y test
# ==============================================================================
X = datos[['tv','radio','periodico']]
y = datos[['ventas']]

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True )                                    


# In[24]:


# Creación del modelo utilizando el modo fórmula (similar a R)
# ==============================================================================
# datos_train = pd.DataFrame(
#                     np.hstack((X_train, y_train)),
#                     columns=['tv', 'radio', 'periodico', 'ventas']
#               )
# modelo = smf.ols(formula = 'ventas ~ tv + radio + periodico', data = datos_train)
# modelo = modelo.fit()
# print(modelo.summary())


# In[27]:


# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo

X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.OLS(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())


# In[ ]:


#El modelo con las variables introducidas como predictores tiene un R2 alto de 0.894 es decir es capaz de explicar
#la variabilidad observada de las vantas en 89,4%. 
#El p-value del modelo es significativo 1.01e-75 indica que la varianza explicada por el modelo es mejor a la esperada por azar (varianza total)

#Acorde al p-value obtenido del coeficiente periodico (0.723) esta variable no es significativa para el modelo. 


# In[28]:


# Se entrena de nuevo el modelo, pero esta vez excluyendo el predictor periodico.
# Creación del modelo utilizando matrices
# ==============================================================================
# Se ELIMINA LA COLUMNA periodico del conjunto de train y test

X_train = X_train.drop(columns = 'periodico')
X_test = X_test.drop(columns = 'periodico')

# A la matriz de predictores se le tiene que añadir una columna de 1s para el
# intercept del modelo

X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.OLS(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())


# In[29]:


#Intervalos de confianza de los coeficientes
# ==============================================================================
intervalos_ci = modelo.conf_int(alpha=0.05)
intervalos_ci.columns = ['2.5%','97,5%']
intervalos_ci


# In[30]:


#Diagnóstico de los resíduos
# Diagnóstico errores (residuos) de las predicciones de entrenamiento
# ==============================================================================

y_train = y_train.flatten()
prediccion_train = modelo.predict(exog = X_train)
residuos_train = prediccion_train - y_train 


# In[32]:


#Inspección visual
# Gráficos
# ==============================================================================
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                'k--', color = 'black', lw=2)
axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
axes[0, 0].set_xlabel('Real')
axes[0, 0].set_ylabel('Predicción')
axes[0, 0].tick_params(labelsize = 7)
#================================================================================
axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
axes[0, 1].set_xlabel('id')
axes[0, 1].set_ylabel('Residuo')
axes[0, 1].tick_params(labelsize = 7)

sns.histplot(
    data    = residuos_train,
    stat    = "density",
    kde     = True,
    line_kws= {'linewidth': 1},
    color   = "firebrick",
    alpha   = 0.3,
    ax      = axes[1, 0]
)
#=================================================================================
axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
                     fontweight = "bold")
axes[1, 0].set_xlabel("Residuo")
axes[1, 0].tick_params(labelsize = 7)


sm.qqplot(
    residuos_train,
    fit   = True,
    line  = 'q',
    ax    = axes[1, 1], 
    color = 'firebrick',
    alpha = 0.4,
    lw    = 2
)
axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
axes[1, 1].tick_params(labelsize = 7)

axes[2, 0].scatter(prediccion_train, residuos_train,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
axes[2, 0].set_xlabel('Predicción')
axes[2, 0].set_ylabel('Residuo')
axes[2, 0].tick_params(labelsize = 7)

# Se eliminan los axes vacíos
fig.delaxes(axes[2,1])

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");


# In[ ]:


#Los residuos no parecen distribuirse de forma aleatoria en torno a cero,sin mantener aproximadamente la misma variabilidad a lo largo del eje X
# Este patrón a que no se distribuyan normalmente y posibles problemas de hetorecadisticidad.


# In[ ]:


#Test de normalidad
"""Se comprueba si los residuos siguen una distribución normal empleando dos test estadísticos: Shapiro-Wilk test 
y D'Agostino's K-squared test. Este último es el que incluye el summary de statsmodels bajo el nombre de Omnibus.

En ambos test, la hipótesis nula considera que los datos siguen una distribución normal, por lo tanto, si el p-value 
no es inferior al nivel de referencia alpha seleccionado, no hay evidencias para descartar que los datos se distribuyen de forma normal
"""


# In[33]:


# Normalidad de los residuos Shapiro-Wilk test
# ==============================================================================
shapiro_test = stats.shapiro(residuos_train)
shapiro_test


# In[34]:


# Normalidad de los residuos D'Agostino's K-squared test
# ==============================================================================
k2, p_value = stats.normaltest(residuos_train)
print(f"Estadítico= {k2}, p-value = {p_value}")


# In[ ]:


#Ambos test muestran claras evidencias para rechazar la Hipótesis Nula de que los datos se distribuyen de forma normal 
#(p-value << 0.01). Cuando los errores no son normales, los intervalos y las pruebas de hipótesis no son exactas y pueden llegar a ser inválidas


# In[35]:


#Predicciones
#Una vez entrenado el modelo, se pueden obtener predicciones para nuevos datos. 
#Los modelos de statsmodels permiten calcular los intervalos de confianza asociados a cada predicción.

# Predicciones con intervalo de confianza 
# ==============================================================================
predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
predicciones.head(4)


# In[36]:


# Error de test del modelo 
# ==============================================================================
X_test = sm.add_constant(X_test, prepend=True)
predicciones = modelo.predict(exog = X_test)
rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")


# In[ ]:


#Interpretación
#El modelo de regresión lineal múltiple:

#  ventas = 2.9004 + 0.0456tv + 0.1904radio

"""El modelo de regresion lineal multiple es capaz de explicar el 89.4% de la variabilidad observada en las ventas (R-squared= 0.894),
El test F es significativo 3.69e-77, lo que quiere decir que el modelo es capaz de explicar la varianza en las ventas 
mejor de lo esperado por azar. Los test estadisticos confirman para cada variable confirman que la tv y radio estan con la cantidad de ventas 
y contribuye al modelo

No se satisfacen las condiciones de normalidad, por lo que los intervalos de confianza estimados para los coeficientes
y las predicciones NO son fiables.

El error RSME de test es de 1.696. Es decir las predicciones se alejan en promedio 1.696 unidades del valor real
"""


# In[ ]:


#Interacción entre predictores
"""https://www.cienciadedatos.net/documentos/py10-regresion-lineal-python.html
. Sin embargo, esto no tiene por qué ser necesariamente así, puede existir interacción entre los predictores de forma que, el efecto de cada uno de ellos sobre la variable respuesta, depende en cierta medida del valor que tome el otro predictor.

Tal y como se ha definido previamente, un modelo lineal con dos predictores sigue la ecuación:

y=β0+β1x1+β2x2+ϵ
 
Acorde a esta definición, el incremento de una unidad en el predictor  x1  produce un incremento promedio de la variable y de  β1 . Modificaciones en el predictor  x2  no alteran este hecho, y lo mismo ocurre con  x2  respecto a  x1 . Para que el modelo pueda contemplar la interacción entre ambos, se introduce un tercer predictor, llamado interaction term, que se construye con el producto de los predictores  x1  y  x2 .

y=β0+β1x1+β2x2+β3x1x2+e
 
La reorganización de los términos resulta en:

y=β0+(β1+β3x2)x1+β2x2+e
 
El efecto de  x1  sobre  y  ya no es constante, sino que depende del valor que tome  x2 .
"""

