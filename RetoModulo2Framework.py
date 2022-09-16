# ==============================================================================================
# Autor: Juan Carlos Varela Tellez
# Fecha de inicio: 09/09/2022
# Fecha de finalizacion: 09/09/2022
# ==============================================================================================
#
# ==============================================================================================
# En caso de no tener las bibliotecas necesarias, utilizar los siguientes comandos:
# python -m pip install numpy
# python -m pip install pandas
# python -m pip install seaborn
# python -m pip install matplotlib
# python -m pip install scikit-learn
# ==============================================================================================
#
# ==============================================================================================
# Las apoplejias son un evento cuando el suministro de sangre al cerebro se ve interrumpida,
# causando en falta de oxigeno, daño cerebral y perdida de funciones tanto motoras como
# mentales.
# Globalmente, 1 de cada 4 adultos mayores de 25 años va a tener una apoplejia en su vida.
# 12,2 millones de personas tendra su primer apoplejia en este año, y 6.5 millones mas
# moriran como resultado de esta. Mas de 110 millones de personas han tenido una apoplejia. (1)
#
# Este codigo tiene como objetivo analizar datos para poder predecir que personas son mas
# propensas a tener una apoplejia y asi poder evitar secuelas y bajar estas estadisticas.
#
# (1) https://www.world-stroke.org/world-stroke-day-campaign/why-stroke-matters/learn-about-stroke#:~:text=Globally%201%20in%204%20adults,the%20world%20have%20experienced%20stroke.
# ==============================================================================================
#
# ==============================================================================================
# Este codigo es una continuacion directa al repositorio https://github.com/JuanVaTe/RetoModulo2,
# asi que se recomienda revisarlo antes de continuar con este codigo.
# ==============================================================================================
#
# ==============================================================================================
# Debido a que nuestros modelos de regresion logistica no fueron lo suficientemente complejos
# para poder dar una prediccion precisa, vamos a utilizar una biblioteca que nos da acceso a
# herramientas y modelos prehechos que nos ayudaran de forma importante con nuestro problema.
# ==============================================================================================

# Importamos librerias

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Ahora leemos nuestro data-set
# Fuente: https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset

stroke_data = pd.read_csv("Data/full_data.csv")

# Primero necesitamos saber que contiene nuestro data-set y sus contenidos, asi que
# sacaremos diferentes metricas para mas adelante poder hacer decisiones mas
# informadas

print(stroke_data.head())

print(stroke_data.info())

# Podemos observar que el data-set tiene 11 caracteristicas, 10 siendo variables independientes
# y 1 siendo la variable dependiente.
# Asimismo, es necesario ver la documentacion para saber que significa cada cosa:
# https://github.com/JuanVaTe/RetoModulo2Framework/blob/main/Images/documentation.png

# Como se hizo en el codigo pasado, vamos a eliminar las filas que cuentan con valores
# nulos. En este caso, no hay ningun valor nulo per se, sin embargo, tenemos casillas
# en la caracteristica de 'smoking_status' que tienen el valor 'Unknown', y esto
# en el contexto de este problema lo podemos contar como un valor nulo.
# Y ya que estamos limpiando, vamos a cambiar la columna de 'Residence_type' a
# 'residence_type' para tener un formato

stroke_data_clean = stroke_data[stroke_data['smoking_status'] != 'Unknown'].reset_index(drop=True).rename(columns={'Residence_type': 'residence_type'})

# El siguiente paso que tenemos que hacer es un analisis estadistico de este data-set,
# sin embargo, no se puede hacer de forma correcta debido a las variables cualitativas.
# Necesitamos cuantificar estas variables. Pandas tiene una funcion que nos va a servir mucho
# en este caso.

# Cuantificamos gender
dummy_gender = pd.get_dummies(stroke_data_clean['gender'])
# Cuantificamos ever_married
dummy_married = pd.get_dummies(stroke_data_clean['ever_married'], prefix='ever_married')
# Cuantificamos work_type
dummy_work_type = pd.get_dummies(stroke_data_clean['work_type'], prefix='work_type')
# Cuantificamos residence_type
dummy_residence_type = pd.get_dummies(stroke_data_clean['residence_type'], prefix='residence_type')
# Cuantificamos smoking_status
dummy_smoking_status = pd.get_dummies(stroke_data_clean['smoking_status'], prefix='smoking_status')

# Se concatena al data-set
stroke_data_clean = pd.concat([stroke_data_clean, dummy_gender, dummy_married, dummy_work_type, dummy_residence_type, dummy_smoking_status], axis=1)

# Por ultimo se eliminan las columnas redundantes
stroke_data_clean = stroke_data_clean.drop(['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status'], axis=1)

# Con los datos cuantificados, ahora podemos hacer nuestro analisis estadistico y encontrar
# correlaciones entre nuestras caracteristicas y nuestra variable dependiente.

correlation = stroke_data_clean.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlation, annot=True)
plt.show()

# Gracias a la grafica podemos apreciar que el factor mas importante entre todos, al menos
# en un aspecto estadistico e individual, es la edad, seguido por la hipertension y enfermedades
# del corazon y el nivel de glucosa.

# Ahora podemos separar nuestro data_set en variables independientes y variable dependiente:
data_x = stroke_data_clean.drop(['stroke'], axis=1)
data_y = stroke_data_clean['stroke']

# Para el preprocesamiento de datos, vamos a quedarnos con 2 data-sets de variables independientes:
# - Data-set con solamente las caracteristicas mas correlacionadas con la apopplejia
# - Data-set con todas las caracteristicas
# Esto es para tener mas espacio de experimentacion al momento de utilizar los
# modelos de machine learning.

data_x_correlation = data_x[['age', 'hypertension', 'heart_disease', 'avg_glucose_level']]

# Por ultimo, vamos a escalar todos los datos. Esto es para ayudar a
# que nuestros modelos encuentren de una forma mas facil la convergencia de la funcion de
# costo.

escalador_all = StandardScaler()
escalador_all.fit(data_x)
data_x_scaled = pd.DataFrame(escalador_all.transform(data_x))

escalador_correlation = StandardScaler()
escalador_correlation.fit(data_x_correlation)
data_x_correlation_scaled = pd.DataFrame(escalador_correlation.transform(data_x_correlation))

# Ahora vamos a modularizar los datos.
# Por buena practica es necesario tener 3 modulos: entrenemiento, validacion y pruebas.
# Por esta ocasion, el modulo de pruebas tendra 5 filas, y sera un subconjunto del
# modulo de validacion.

train_x_all, test_x_all, train_y_all, test_y_all = train_test_split(data_x_scaled, data_y, random_state=0)

train_x_corr, test_x_corr, train_y_corr, test_y_corr = train_test_split(data_x_correlation_scaled, data_y, random_state=0)

# Ahora podemos empezar a probar modelos.

# Empecemos con un modelo sencillo, que es el de regresion logistica.

logistic_regression = LogisticRegression(random_state=0)
logistic_regression.fit(train_x_all, train_y_all)

print("Regresion logistica ===============================================")
print("Puntaje de entrenamiento:", logistic_regression.score(train_x_all, train_y_all))
print("Puntaje de validacion:", logistic_regression.score(test_x_all, test_y_all))
print("===============================================\n")

# Curiosamente, desde la primera iteracion de experimentacion del modelo
# dio un resultado muy alto. Se intento con ambos data-sets, cambiando el
# optimizador, diferentes hiperparametros del modelo, pero al final,
# el default fue el que mayor puntaje tuvo.

# Sigamos con arboles de decision

decision_tree = DecisionTreeClassifier(random_state=0,
                                       max_depth=12)
decision_tree.fit(train_x_corr, train_y_corr)

print("Arbol de decision sin podar ======================================")
print("Puntaje de entrenamiento:", decision_tree.score(train_x_corr, train_y_corr))
print("Puntaje de validacion:", decision_tree.score(test_x_corr, test_y_corr))
print("Profundidad maxima:", decision_tree.tree_.max_depth)
print("===============================================\n")

pruning_data = decision_tree.cost_complexity_pruning_path(train_x_corr, train_y_corr)

best_alpha = 0
best_score = 0

# Probando todas las alfas
for alpha in pruning_data.ccp_alphas:
    tree = DecisionTreeClassifier(random_state=0,
                                  max_depth=12,
                                  ccp_alpha=alpha)
    tree.fit(train_x_corr, train_y_corr)

    if tree.score(test_x_corr, test_y_corr) > best_score:
        best_score = tree.score(test_x_corr, test_y_corr)
        best_alpha = alpha

# Nuevo arbol con la mejor alfa
decision_tree = DecisionTreeClassifier(random_state=0,
                                       max_depth=12,
                                       ccp_alpha=best_alpha)
decision_tree.fit(train_x_corr, train_y_corr)

print("Arbol de decision podado ======================================")
print("Puntaje de entrenamiento:", decision_tree.score(train_x_corr, train_y_corr))
print("Puntaje de validacion:", decision_tree.score(test_x_corr, test_y_corr))
print("Mejor alfa:", best_alpha)
print("=======================================================\n")

# En este ejemplo, lo mejor fue utilizar el data-set con solamente los datos correlacionados.
# Esto ayuda a generalizar mejor, ya que los arboles de decision tienden a tener
# overfitting.
# Algo que me parece muy raro es el hecho de que se obtiene el mismo puntaje de
# validacion que la regresion logistica, es casi como si los mismos datos nos dijeran que esa es
# la mejor puntacion que podemos obtener

# Lo siguiente es bosques aleatorios

random_forest = RandomForestClassifier(random_state=0,
                                       n_estimators=1000,
                                       max_depth=11,
                                       min_samples_leaf=5)
random_forest.fit(train_x_all, train_y_all)

print("Bosque aleatorio ===============================================")
print("Puntaje de entrenamiento:", random_forest.score(train_x_all, train_y_all))
print("Puntaje de validacion:", random_forest.score(test_x_all, test_y_all))
print("===============================================\n")

# En este caso fue mejor utilizar el data-set con todos los datos.
# El bosque aleatorio tiene medidas para contrarrestar el overfitting, es por esto
# que utilizar este data-set obtiene mejores resultados
# Una vez mas aparecio este puntaje, el mismo que los otros modelos
# anteriores, sacados con data-sets diferentes. Es bastante curioso esto.
# Por ultimo quisiera remarcar que este bosque aleatorio no fue el que tuvo
# mejor puntaje en su entrenamiento, sin embargo, se prefiere que el puntaje
# de la validacion sea mas alto ya que esto indica que el modelo generaliza de una mejor
# manera, tanto asi que predice mejor con datos nuevos que con datos
# que utilizo para entrenar.

# Por ultimo quedan las redes neuronales.

neural_network_full = MLPClassifier(random_state=0,
                                    hidden_layer_sizes=(8, 8),
                                    learning_rate='adaptive',
                                    max_iter=10000)
neural_network_full.fit(train_x_all, train_y_all)

print("Red neuronal con data-set completo ===========================")
print("Puntaje de entrenamiento:", neural_network_full.score(train_x_all, train_y_all))
print("Puntaje de validacion:", neural_network_full.score(test_x_all, test_y_all))
print("===============================================\n")

neural_network_corr = MLPClassifier(random_state=0,
                                    hidden_layer_sizes=(8, 8),
                                    learning_rate='adaptive',
                                    max_iter=10000)
neural_network_corr.fit(train_x_corr, train_y_corr)

print("Red neuronal con data-set correlacional =======================")
print("Puntaje de entrenamiento:", neural_network_corr.score(train_x_corr, train_y_corr))
print("Puntaje de validacion:", neural_network_corr.score(test_x_corr, test_y_corr))
print("===============================================\n")

# Por la naturaleza de una red neuronal, utilizar uno u otro data-set no
# tiene tanta diferencia como los otros modelos. Esto es porque este cambio
# de pesos para cada caracteristica es algo intrinseco en la red neuronal,
# mejorando con cada iteracion al darle mas importancia a las caracteristicas
# que la necesitan.
# No hay mejor demostracion de esto que el hecho de que ambas rede neuronales
# hayan llegado al mismo puntaje de validacion y un puntaje extremadamente
# similar en su puntaje de entrenamiento, aunque en algunos casos tambien quedo
# igual.
# Es por esta razon por la cual quise dejar ambas redes neuronales, para demostracion
# de como se comporta una red neuronal.

# Por ultimo, vamos a hacer unas comparaciones de modelos con matrices de confusion
# ya que ademas de dar metricas de rendimiento, tambien nos sirve para ver cuantas
# predicciones fueron correctas y cuantas no.

# Definicion de funcion para sacar las metricas de rendimiento
def metricas_rendimiento(matriz_confusion):
    exactitud = (matriz_confusion[0][0] + matriz_confusion[1][1]) / (
                matriz_confusion[0][0] + matriz_confusion[0][1] + matriz_confusion[1][0] + matriz_confusion[1][1])

    try:
        precision = matriz_confusion[0][0] / (matriz_confusion[0][0] + matriz_confusion[1][0])
    except:
        precision = 0

    exhaustividad = matriz_confusion[0][0] / (matriz_confusion[0][0] + matriz_confusion[0][1])

    try:
        puntaje_F1 = (2 * precision * exhaustividad) / (precision + exhaustividad)
    except:
        puntaje_F1 = 0

    return exactitud, precision, exhaustividad, puntaje_F1

# Listado de modelos para ciclar la obtencion de metricas y matrices de confusion

models = [['Regresion Logistica', 'all', logistic_regression],
          ['Arbol de decision', 'corr', decision_tree],
          ['Bosque aleatorio', 'all', random_forest],
          ['Red neuronal con data-set completo', 'all', neural_network_full],
          ['Red neuronal con data-set correlacioal', 'corr', neural_network_corr]]

for trio in models:
    if trio[1] == 'all':
        conf_matrix = confusion_matrix(test_y_all, trio[2].predict(test_x_all))
    else:
        conf_matrix = confusion_matrix(test_y_corr, trio[2].predict(test_x_corr))

    acc, prec, recall, F1_score = metricas_rendimiento(conf_matrix)

    print("=============================================")
    print(f"Metricas de rendimiento para modelo de {trio[0]}")
    print("Matriz de confusion:")
    print(conf_matrix)
    print(f"Exactitud     : {acc}")
    print(f"Precision     : {prec}")
    print(f"Exhaustividad : {recall}")
    print(f"Puntaje F1    : {F1_score}")
    print("=============================================\n")

# Aqui es donde se acaba la magia.
# La razon por la cual nuestros modelos se parecian tanto es porque todos nuestros modelos
# se inclinan demasiado a decir que no hay peligro de apoplejia, tanto asi que para estos
# modelos no es posible que a alguien le de una apoplejia.
# Sin embargo, los modelos utilizados aun nos dejan con el aprendizaje de una biblioteca muy
# completa. En un futuro cercano, se intentara utilizar nuevamente esta biblioteca
# para poder hacer predicciones mucho mas certeras.

# Por ultimo vamos a hacer unas predicciones con la ultima red neuronal

# Ya que el modelo mas complejo fue este ultimo, vamos a hacer unas predicciones para
# verificar estos resultados

for i in range(15, 20):
    print("Prediccion numero", i - 14, "======================")
    print("Datos:", test_x_all.iloc[i])
    print("Prediccion:", neural_network_full.predict(pd.DataFrame(test_x_all.iloc[0]).transpose()))
    print("Realidad:", test_y_all.iloc[i])
    print("=============================================")
