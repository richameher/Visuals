import numpy as np
import matplotlib.pyplot as plt

def not_allowed(*args, **kwargs):
    raise RuntimeError("You called an illegal function.")

import scipy.linalg as sla
for attr in dir(sla):
    setattr(sla, attr, not_allowed)
import numpy.linalg as la
la.solve = not_allowed

# Number of points in a side
n = 30

# Capacitor plate positions
plate_xpos = [9,19]
plate_ymin = 9
plate_ymax = 19

def get_A():
    A = np.zeros((n**2, n**2))
    for i in range(n**2):
        x = i % n
        y = i // n
        if y >= plate_ymin and y <= plate_ymax and x in plate_xpos:
            A[i,i] = 4
            continue
        A[i,i] = -4
        if x < n - 1: A[i,i+1] = 1
        if x > 0:     A[i,i-1] = 1
        if y < n - 1: A[i,i+n] = 1
        if y > 0:     A[i,i-n] = 1
    return A

def get_b():
    sp = np.linspace(0, 1, n)
    xs, ys = np.meshgrid(sp, sp)
    ys_in_range = (ys >= sp[plate_ymin]) & (ys <= sp[plate_ymax])
    ones = np.ones(xs.shape)
    zeros = np.zeros(xs.shape)
    b =  np.where((xs == sp[plate_xpos[0]]) & ys_in_range, -ones, zeros)
    b += np.where((xs == sp[plate_xpos[1]]) & ys_in_range, ones, zeros)
    return 4 * b.ravel()

A = get_A()
b = get_b()
k = n

def plot_solution(soln):
    plt.figure(figsize=(4,4))
    plt.imshow(soln.reshape((n,n)))
    plt.colorbar()


    # Call this function after you have computed x.
    import numpy as np
    sha=np.shape(A)
    shap=np.shape(A)
    M1 = np.eye(shap[0])
    L = np.eye(shap[0])
    x=np.zeros(shap[0])
    for i in range(1,A.shape[0],1):
        lower=i-1
        upper=i+k
        if(i+k > A.shape[0]):
                    upper=A.shape[0]
        for j in range(i,upper,1):
                ele=-A[j,i-1]/A[i-1,i-1]
                L[j,i-1]=-ele
                for l in range(lower,upper,1):
                    A[j,l]+=ele*A[i-1,l]

    U=np.copy(A)
    c=np.zeros(A.shape[0])
    n=A.shape[0]
    c[0]=b[0]

    for i in range(1,n,1):
        temp=b[i]
        for j in range(0,i,1):
            temp=temp-L[i,j]*c[j]
        c[i]=temp/L[i,i]
    x=np.zeros(U.shape[0])


    for i in range(n-1,-1,-1):
        temp=c[i]
        for j in range(n-1,i,-1):
            temp=temp-U[i,j]*x[j]
        x[i]=temp/U[i,i]

    plot_solution(x)
