# -*- coding: utf-8 -*-
import numpy             as np
from csv       import reader
from pregunta1 import iteraciones

np.random.seed(42)

# Convertir una columna de etiquetas a valores numéricos 
def from_nominal(matrix):
    return np.transpose(np.matrix(np.unique(np.asarray(matrix),return_inverse=True)[1]))

# Lectura de archivo, conversión de tipos y randomize de filas
file = open("AmesHousing.csv","r")
data = list(reader(file, delimiter=',', quotechar='\"'))
data = np.matrix(data)


nominal_cols = [3 ,6 ,7 ,8 ,9, 10,11,12,13,14
               ,15,16,17,22,23,24,25,26,28,29
               ,30,31,32,33,34,36,40,41,42,43
               ,54,56,58,59,61,64,65,66,73,74
               ,75,79,80]
for col in nominal_cols:
    data[:,col] = from_nominal(data[:,col])

# Separando header de datos y convirtiendo a float
data   = data[1:]
header = data[0]
data[data==''] = "NaN"
data = data.astype(np.float)

np.random.shuffle(data)

# Conjunto de entrenamiento y de prueba sin la columna de id's
N = data.shape[0]
n = data.shape[0]*0.8

# Removiendo la primera columna de ids y agregando un vector de 1's para
# poder realizar la regresión lineal
training_d = np.column_stack((np.ones((n,1)),data[:n,:]))
# test_d     = np.column_stack((np.ones((N-n+1,1)),data[n:,:]))

iterations = 200
theta      = np.ones((1,data.shape[1]))

# import pdb;  pdb.set_trace()

res = iteraciones(training_d[:,:-1].T
                 ,training_d[:,-1]
                 ,n
                 ,theta
                 ,0.3
                 ,iterations)

# RECORDAR REINICIAR THETA EN PREGUNTA 2