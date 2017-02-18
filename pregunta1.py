# -*- coding: utf-8 -*-
# Proyecto 1 - Primera Pregunta
# Universidad Simón Bolívar, 2017.
# Author: Carlos Farinha
# Last Revision: 17/02/17
# Modified by: Carlos Farinha, Javier López

# Implemente el algoritmo de Descenso del Gradiente para resolver una Regresion Lineal Multiple
# en el lenguaje de programacion de su preferencia entre C, C++, Java o Python.

import numpy as np

# import operator as op
# import matplotlib.pyplot as plt
# import functools
# foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)

# Funcion de Gradient Descent
def gradientDescent(xi,yi,n,theta,alpha):
	#print(theta)
	#theta=np.matrix(theta)
	#xi=np.matrix(xi)
	#yi=np.matrix(yi)
	hoxy = np.dot(theta,xi)-yi
	# try:
	# 	print('np')
	# 	print(theta.T.shape)
	# 	print(xi.shape)
	# 	print(yi.shape)
	# 	print(hoxy.shape)
	# except:
	# 	print('np2')
	# 	try:
	# 		print('(%d,%d)'%  (len(theta), len(theta[0])))
	# 	except:
	# 		print('(%d,%d)'%  (len(theta), 1))
	# 	try:
	# 		print('(%d,%d)'%  (len(xi), len(xi[0])))
	# 	except:
	# 		print('(%d,%d)'%  (len(xi), 1))
	# 	try:
	# 		print('(%d,%d)'%  (len(yi), len(yi[0])))
	# 	except:
	# 		print('(%d,%d)'%  (len(yi), 1))
	# 	try:
	# 		print('(%d,%d)'%  (len(hoxy), len(hoxy[0])))
	# 	except:
	# 		print('(%d,%d)'%  (len(hoxy), 1))
	# print('hola')

	try:
		aux = xi.T
	except:
		aux = xi
	#print(aux[1,1])
	#hoxyx = np.dot(hoxy,aux)
	hoxyx = np.multiply(hoxy,aux)
	dojo =[]	
	for each in hoxyx:
		dojo.append(np.sum(each))
	fdojo = np.divide(dojo,float(n))
	newtheta = np.subtract(theta,np.multiply(alpha,fdojo))
	return(newtheta)

# Funcion de costo J(0)
def costFuntionJ(xi,yi,theta,n):
	hoxy = np.dot(theta,xi)-yi
	j = float(np.sum(np.power(hoxy,2)))/float(2*n)
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
		if np.sum((np.subtract(newtheta,theta))) == 0:
			return([newtheta,jos,xx])
		theta = newtheta
		i = i+1
	return([theta,jos,xx])
