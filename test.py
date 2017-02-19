import numpy as np
import random

def normalize(matrix,columns=None):
    mean = matrix.mean(0)
    std  = matrix.std(0)
    
    if not columns:
        columns = range(0,matrix.shape[1])
    for i in columns:
        print(type(std[i]))
        if std[i] != 0:
            matrix[:,i] = (matrix[:,i] - mean[i]) / std[i]
    # else:
    #     matrix = (matrix - mean) / std

    return matrix

def costFuntionJ(xi,yi,theta,n):
    hoxy = np.dot(theta,xi)-yi
    j = float(np.sum(np.power(hoxy,2)))/float(2*n)
    return (j)

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    # print(theta.shape)
    # print(x.shape)
    # print(y.shape)
    # print(type(theta))
    # print(type(x))
    # print(type(y))
    costs = np.zeros((numIterations,1))
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        if i == -1:
            print('shapes1')
            print(y.shape)
            print(hypothesis.shape)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        costs[i]=(float(np.sum( loss * loss)) / (2 * m))
        if not np.isinf(costs[i]):
            print(costs[i])
        # avg gradient per example
        if i == -1:
            print('shapes')
            print(x.T.shape)
            print(loss.shape)
        gradient = np.dot(x.T, loss) / m
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