# Proyecto 1 - Primera Pregunta
# Universidad Simon Bolivar, 2017.
# Author: Carlos Farinha
# Last Revision: 10/02/17
# Modified by: Carlos Farinha

# Implemente el algoritmo de Descenso del Gradiente para resolver una Regresion Lineal Multiple
# en el lenguaje de programacion de su preferencia entre C, C++, Java o Python.

import matplotlib.pyplot as plt
import numpy as np
import functools
import operator as op

foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)

# Funcion de Gradient Descent
def gradientDescent(xi,yi,n,theta,alpha):
	hoxy = np.dot(theta,xi)-yi
	hoxyx = np.multiply(hoxy,xi)
	dojo =[]	
	for each in hoxyx:
		dojo.append(foldl(op.add, 0, each))
	fdojo = np.divide(dojo,float(n))
	newtheta = np.subtract(theta,np.multiply(alpha,fdojo))
	return(newtheta)

# Funcion de costo J(0)
def costFuntionJ(xi,yi,theta,n):
	hoxy = np.dot(theta,xi)-yi
	j = float((foldl(op.add, 0, np.power(hoxy,2))))/float(2*n)
	return (j)

# Normalizacion de datos
def normalizacion(xi,newlines,normal):
	if normal == 1:
		means = []
		stds = []
		for each in xi:
			means.append(np.matrix(each).mean())
			stds.append(np.matrix(each).std())
		xni = np.transpose(np.divide((np.subtract(np.transpose(xi),means)),stds))
		xnormalizada = [[1]*(len(newlines)-1)]
		xnormalizada.extend(xni)
	if normal == 0:
		xnormalizada = [[1]*(len(newlines)-1)]
		xnormalizada.extend(xi)
	return(xnormalizada)

#Iterador de la funcion de gradiente para i iteraciones
def iteraciones(xi,yi,n,theta,alpha,iterations):
	i = 0
	jos = []
	xx = []
	while (i < iterations):
		xx.append(i)
		newtheta = gradientDescent(xi,yi,n,theta,alpha)
		jo = costFuntionJ(xi,yi,newtheta,n)
		jos.append(jo)
		if ((foldl(op.add, 0, np.subtract(newtheta,theta))) == 0):
			return([newtheta,jos,xx])
		theta = newtheta
		i = i+1
	return([theta,jos,xx])
