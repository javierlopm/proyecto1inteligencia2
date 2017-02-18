# -*- coding: utf-8 -*-
# Proyecto 1 - Primera Pregunta
# Universidad Simón Bolívar, 2017.
# Author: Carlos Farinha
# Last Revision: 17/02/17
# Modified by: Carlos Farinha, Javier López

import matplotlib.pyplot as plt
import numpy as np
import functools
import operator as op
import pregunta1 as p

#Variables Globales
iterations = 100


# Pregunta 1.1 
#a)  Muestre la curva de convergencia (J() vs. Iteraciones) para alpha = 0.1
#b)  Realice un scatteplot de los datos junto con la curva que minimiza la funcion de costo

# Lectura del archivo
file = open("x01.txt","r")
lines = file.read().splitlines()
newlines = []
for x in lines:
	newlines.append(filter((lambda y: y != ''),x.split(' ')))
x = []
yi = []
i = 0
while (i < len(newlines)-1):
	l = len(newlines[i])-1
	yi.append(float(newlines[i][l]))
	l = l-1
	temp = []
	while (l > 0) :
		temp.append(float(newlines[i][len(newlines[i])-1-l]))
		l = l-1
	x.append(temp)
	i = i+1
xi = []
xi.extend(np.transpose(x))

#Numero de casos leidos
n = len(yi)

#Inicializacion del vector theta
theta = [0]
theta.extend([1]*(len(xi)))

#Normalizacion de datos
xi = p.normalizacion(xi,newlines,1)

# Iteraciones del proceso
val = p.iteraciones(xi,yi,n,theta,0.1,iterations)
theta = val[0]
jos = val[1]
xx = val[2]

# Impresion de Graficos y resultados 1.1 a)	
plt.plot(xx, jos, label="multi\nline")
plt.title ('1.1 a) Convergencia de J(0) vs iteraciones para alpha = 0.1')
plt.ylabel('Valor de J(0) en la iteracion')
plt.xlabel('Numero de iteraciones')
plt.show()

# Impresion de Graficos y resultados 1.1 b)	
plt.scatter(xi[1], yi, s=10, alpha=0.5)
xx = np.linspace(-1, 8)
plt.plot(xx, np.add(np.dot(theta[1],xx),theta[0]) , label="multi\nline")
plt.title ('1.1 b)  Scatteplot y curva min de funcion de costo')
plt.ylabel('Valor de y ')
plt.xlabel('Valor de x')
plt.show()

################
# Pregunta 1.2 #
################


iterations = 100

# Lectura del archivo
file = open("x08.txt","r")
lines = file.read().splitlines()
newlines = []
for x in lines:
	newlines.append(filter((lambda y: y != ''),x.split(' ')))
x = []
yi = []
i = 0
while (i < len(newlines)-1):
	l = len(newlines[i])-1
	yi.append(float(newlines[i][l]))
	l = l-1
	temp = []
	while (l > 0) :
		temp.append(float(newlines[i][len(newlines[i])-1-l]))
		l = l-1
	x.append(temp)
	i = i+1
xi = []
xi.extend(np.transpose(x))

#Inicializacion del vector theta
theta = [0]
theta.extend([1]*(len(xi)))
m = len(xi)
#Numero de casos
n = len(yi)
xx = range(0,100)
jos =[]
hoxys = []

#Normalizacion de datos
xi = p.normalizacion(xi,newlines,1)

# Iteraciones del proceso
val = p.iteraciones(xi,yi,n,theta,0.1,iterations)
theta = val[0]
jos = val[1]
xx = val[2]
plt.plot(xx, jos, label="multi\nline")
print ("theta for alpha = 0,1 ----> %s" %theta)
print ("xx for alpha = 0,1 ----> %s" %xx)
print("jos : %s"%jos)


for alpha in [0.1,0.3,0.5,0.7,0.9,1]:
    # theta = [0]
    # theta.extend([1]*m)
    theta    = np.ones((1,m+1))
    theta[0] = 0

    val = p.iteraciones(xi,yi,n,theta,alpha,iterations)
    theta = val[0]
    jos   = val[1]
    xx    = val[2]
    plt.plot(xx, jos, label="multi\nline")
    print ("theta for alpha = {} ----> {}".format(alpha,theta))
    # print ("xx    for alpha = {} ----> {}".format(alpha,xx))
    # print ("jos             = %s\n\n"%jos)

# Impresion de Graficos y resultados
#plt.yscale('log',basey=10)
plt.show()


#plt.scatter(xi[1], yi, s=20, alpha=0.5)
xx = np.linspace(-1, 8)
#plt.plot(xx, np.add(np.dot(theta[1],xx),theta[0]) , label="multi\nline")
#plt.show()



