# -*- coding: utf-8 -*-
# Proyecto 1 - Primera Pregunta
# Universidad Simón Bolívar, 2017.
# Authors: Carlos Farinha   09-10270
#          Javier López     11-10552
#          Nabil J. Marquez 11-10683
# Last Revision: 18/02/17

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

#normalizacion_l de datos
xi = p.normalizacion_l(xi,newlines,1)

# Iteraciones del proceso
val = p.gradientDescent(xi,yi,theta,0.1,n,iterations)
theta = val[0]
jos = val[1]
xx = range(iterations)

# Impresion de Graficos y resultados 1.1 a)	
plt.plot(xx, jos, label="alpha = 0.1")
plt.title ('1.1 a) Convergencia de J(0) vs iteraciones para alpha = 0.1')
plt.ylabel('Valor de J(0) en la iteracion')
plt.xlabel('Numero de iteraciones')
plt.legend()
plt.show()

# Impresion de Graficos y resultados 1.1 b)	
plt.scatter(xi[1], yi, s=10, alpha=0.5)
xx = np.linspace(-1, 8)
print(len(theta))
plt.plot(xx, np.add(np.dot(theta[1],xx),theta[0]) , label="Linear Regression")
plt.title ('1.1 b)  Scatteplot y curva min de funcion de costo')
plt.ylabel('Valor de y ')
plt.xlabel('Valor de x')
plt.legend()
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

#normalizacion_l de datos
xi = p.normalizacion_l(xi,newlines,1)

# Iteraciones del proceso
for alpha in [0.1,0.3,0.5,0.7,0.9,1]:
    theta      = np.ones(m+1, dtype=np.float128)
    theta[0] = 0

    val = p.gradientDescent(xi,yi,theta,alpha,n,iterations)
    theta = val[0]
    jos   = val[1]
    xx    = range(iterations)
    plt.plot(xx, jos, label="alpha = %f" % alpha)
    print ("theta for alpha = {} ----> {}".format(alpha,theta))

# Impresion de Graficos y resultados
plt.legend()
plt.show()


xx = np.linspace(-1, 8)



