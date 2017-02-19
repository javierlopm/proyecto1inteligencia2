# -*- coding: utf-8 -*-
# Proyecto 1 - Primera Pregunta
# Universidad Simón Bolívar, 2017.
# Authors: Carlos Farinha   09-10270
#          Javier López     11-10552
#          Nabil J. Marquez 11-10683
# Last Revision: 18/02/17

# Implemente el algoritmo de Descenso del Gradiente para resolver una Regresion Lineal Multiple
# en el lenguaje de programacion de su preferencia entre C, C++, Java o Python.

import numpy as np

# Normalizacion de datos
def normalizacion(matrix,columns=None):
    mean = np.mean(matrix,0)
    std  = np.std(matrix,0)
    
    if not columns:
        columns = range(0,matrix.shape[1])
    for i in columns:
        print(type(std[i]))
        if std[i] != 0:
            matrix[:,i] = (matrix[:,i] - mean[i]) / std[i]
    return matrix


# Funcion de Gradient Descent
def gradientDescent(x, y, theta, alpha, m, numIterations):
    #So we may use both numpy.arrays and lists
    if (type(theta) is list):
        theta = np.array(theta)
    if (type(x) is list):
        x = np.array(x).T
    if (type(y) is list):
        y = np.array(y)

    jos = np.zeros((numIterations,1))   #Costs
    xTrans = x.T
    for i in range(0, numIterations):
        hoxy = np.dot(x, theta) - y   #Ho-y, Loss
        jos[i]=(float(np.sum(np.power(hoxy,2))) / (2 * m))  #Costs
        gradient = np.dot(xTrans, hoxy) / m
        theta = theta - alpha * gradient #Update theta
    return (theta,jos)

# Funcion de costo J(0)
def costFuntionJ(xi,yi,theta,n):
	hoxy = np.dot(theta,xi)-yi
	j = float(np.sum(np.power(hoxy,2)))/float(2*n)
	return (j)

# Normalizacion de datos para listas
def normalizacion_l(xi,newlines,normal):
	if normal == 1:
		means = []
		stds = []
		for each in xi:
			means.append(np.matrix(each).mean())
			stds.append(np.matrix(each).std())
		xni = np.transpose(np.divide((np.subtract(np.transpose(xi),means)),stds))
		xnormalized = [[1]*(len(newlines)-1)]
		xnormalized.extend(xni)
	if normal == 0:
		xnormalized = [[1]*(len(newlines)-1)]
		xnormalized.extend(xi)
	return(xnormalized)

