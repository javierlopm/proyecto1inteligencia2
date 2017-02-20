# -*- coding: utf-8 -*-
import numpy     as np
from csv         import reader
from pregunta1   import *
from collections import Counter
from decimal     import Decimal
np.random.seed(42)


# Convertir una columna de etiquetas a valores numéricos y colocar la moda en vacios
def from_nominal(matrix):
    uniques = np.unique(np.asarray(matrix),return_inverse=True)[1]
    mode    = Counter(uniques.tolist()).most_common()[0][0]
    res     = np.transpose(np.matrix(uniques))
    # res[matrix==''] = mode
    return res

# Lectura de archivo, conversión de tipos y randomize de filas
file = open("AmesHousing.csv","r")
data = list(reader(file, delimiter=',', quotechar='\"'))
data = np.matrix(data)


nominal_cols = [3 ,6 ,7 ,8 ,9, 10,11,12,13,14
               ,15,16,17,22,23,24,25,26,28,29
               ,30,31,32,33,34,36,40,41,42,43
               ,54,56,58,59,61,64,65,66,73,74
               ,75,79,80]

not_nominals = filter(lambda x:not x in nominal_cols,range(0,83))

# Completar valores vacíos y modificar nominales
for col in nominal_cols:
    data[1:,col] = from_nominal(data[1:,col])

# Remover header, cambiar vacíos en columnas no nominales por media
data   = data[1:]
header = data[0]
data[data==''] = "NaN"
data           = data.astype(np.float)
data = np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=0), data)
# import pdb; pdb.set_trace()


np.random.shuffle(data)

# Conjunto de entrenamiento y de prueba sin la columna de id's
N = data.shape[0]
n = int(data.shape[0]*0.8)

# Normalizando y agregando columna de 1's
training_d = data[:n,:]
mu,std     = normalizacion(training_d)
training_d = np.column_stack((np.ones((training_d.shape[0],1)),training_d))


print("Model training and evaluations\n")
# Training model
iterations = 113
x     = np.array(training_d[:,:-1].tolist(), dtype=np.float128)
y     = training_d[:,-1].T.tolist()
y     = np.array(y[0], dtype=np.float128)
theta = np.ones(data.shape[1], dtype=np.float128)
theta,costs = gradientDescent(x,y,theta,0.1,n,iterations)
print("Training set cost: {}".format(costFuntionJ(x,y,theta,N-n+1)))
eval_model(theta,x,y)
print("\n")

# Checking cost over testing set
test_d  = data[n:,:]
normalizacion(test_d,mu,std)
test_d = np.column_stack((np.ones((test_d.shape[0],1)),test_d))
x       = np.array(test_d[:,:-1].tolist(), dtype=np.float128)
y       = test_d[:,-1].T.tolist()
y       = np.array(y[0], dtype=np.float128)
print("Testing set cost : {}".format(costFuntionJ(x,y,theta,N-n+1)))
eval_model(theta,x,y)
