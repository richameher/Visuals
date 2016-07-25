import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

m = data.shape[0]
t= np.zeros(m)
y=np.zeros(m)
mu=1
x=xi
for i in range(0,m):
    t[i] = data[i][0]
    y[i] = data[i][1]


def f(x):
    return (x[0] * np.exp(x[1] * t)*np.sin(x[2]*t+x[3])+x[4])
def jac(x):
    return np.array([
        -np.exp(x[1] * t)*np.sin(x[2]*t+x[3]),
        -x[0]*t*np.exp(x[1] * t)*np.sin(x[2]*t+x[3]),
        -x[0]*t*np.exp(x[1] * t)*np.cos(x[2]*t+x[3]),
        -x[0]*np.exp(x[1] * t)*np.cos(x[2]*t+x[3]),
            -np.ones(m)
        ]).T
def phi(x):
    return (0.5*(res(x).T@res(x)))
def res(x):
    return (y - (x[0] * np.exp(x[1] * t)*np.sin(x[2]*t+x[3])+x[4]))

def newton_lev(x):
    val = (x+la.solve(jac(x).T@jac(x)+mu*np.eye(5),-jac(x).T@res(x)))
    return val

temp = []
temp.append(xi)
k = 0
while(phi(temp[k])>tol):
    x_next = newton_lev(temp[k])
    if (phi(x_next)<phi(temp[k])):
        temp.append(x_next)
        mu = mu*0.5
    else:
        mu = 2*mu
        temp.append(temp[k])
    k = k+1

mu = 2*mu
x = temp[k]
phi = phi(x)


plt.plot(t,f(x),'r',label = 'sinosoidal function')
plt.plot(t,y,'o',label = 'data')
plt.xlabel('$t_i$')
plt.ylabel('$y_i $')
