# -*- coding: utf-8 -*-

import numpy as np
from csv import reader

# Convertir una columna de etiquetas a valores numéricos 
def from_nominal(matrix):
    return np.transpose(np.matrix(np.unique(np.asarray(matrix),return_inverse=True)[1]))

file = open("AmesHousing.csv","r")
data = list(reader(file, delimiter=',', quotechar='\"'))

nominal_cols = [3 ,6 ,7 ,8 ,9, 10,11,12,13,14
               ,15,16,17,22,23,24,25,26,28,29
               ,30,31,32,33,34,36,40,41,42,43
               ,54,56,58,59,61,64,65,66,73,74
               ,75,79,80]

nominal_values = [[] for col in nominal_cols]

data = np.matrix(data)


for col in nominal_cols:
    data[1:,col] = from_nominal(data[1:,col])

