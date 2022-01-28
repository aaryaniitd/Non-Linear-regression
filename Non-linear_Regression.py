# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot as plt

# Optimization module in scipy
from scipy import optimize

# tells matplotlib to embed plots within the notebook
%matplotlib inline

data = np.loadtxt('C:\Semester 4\APL405\Data Files\prob1data.txt',delimiter = ',')
#print(data)
X = data[0]
y = data[1]
#print(X[50])
#print(y[50])
#print(len(X),len(y))
plt.figure()
plt.plot(X,y,'o')

def predicted(weights,params):
    temp = []
    for i in range(len(params)):
        temp.append(np.dot(weights,params[i]))
    return temp

def gradientDescent(x,y,params,weights,alpha):
    for j in range(len(weights)):
        s = 0
        for i in range(len(x)):
            s += (np.dot(params[i],weights)-y[i])*x[i]
        weights[j] -= alpha*s/len(x)
    return weights
def costFunction(weights,params,y):
    m = len(y)
    temp = 0
    for i in range(m):
        temp += (np.dot(params[i],weights)-y[i])**2
    return temp/2/m
def plotter(x,params,y,weights,iterations,alpha):
    temp1 = []
    temp2 = []
    j = 0
    for i in range(2,iterations):
        j = costFunction(gradientDescent(x,y,params,weights,alpha),params,y)
        temp1.append(j)
        temp2.append(i)
    plt.figure()
    plt.plot(temp2,temp1)
    plt.title('J vs Iterations')
    return temp1[-1]

#setting up initial values
weights = [0,13,-4]
params = [[1,x,x**2] for x in X]
alpha = 0.01

print("Initial Cost Function:",costFunction(weights,params,y)) #initial cost function   
print("Initial Weights:",weights)

# Altering the value of alpha and checking different values of cost function alongwith plotting J vs Iterations
print("J for alpha = 0.5:",plotter(X,params,y,[0,13,-4],50,0.5))
print("J for alpha = 0.1:",plotter(X,params,y,[0,13,-4],50,0.1))
print("J for alpha = 0.05:",plotter(X,params,y,[0,13,-4],50,0.05))
print("J for alpha = 0.01:",plotter(X,params,y,[0,13,-4],50,0.01))
print("J for alpha = 0.001:",plotter(X,params,y,[0,13,-4],50,0.001))

#plotting predicted values alongwith original ones
plt.figure()
plt.plot(X,y,'o')
plt.plot(X,predicted(weights,params),'o')
