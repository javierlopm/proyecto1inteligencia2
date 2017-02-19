import numpy as np
import random

def costFuntionJ(xi,yi,theta,n):
    hoxy = np.dot(theta,xi)-yi
    j = float(np.sum(np.power(hoxy,2)))/float(2*n)
    return (j)

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    print(theta.shape)
    print(x.shape)
    print(y.shape)
    print(type(theta))
    print(type(x))
    print(type(y))
    costs = np.ones(numIterations)
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        if i == 0:
            print('shapes1')
            print(y.shape)
            print(hypothesis.shape)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        costs[i]=(float(np.sum(np.power(loss,2))) / (2 * m))
        # avg gradient per example
        if i == 0:
            print('shapes')
            print(xTrans.shape)
            print(loss.shape)
        gradient = np.dot(xTrans, loss) / m
        #print("%s grdient" % str(gradient.shape))
        # update
        theta = theta - alpha * gradient
    return (theta,costs)


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
if __name__=="__main__" :
    data = genData(100, 25, 10)
    x, y = data
    m, n = np.shape(x)
    numIterations= 100000
    alpha = 0.0005
    theta = np.ones(n)

    theta = gradientDescent(x, y, theta, alpha, m, numIterations)
    print(theta)