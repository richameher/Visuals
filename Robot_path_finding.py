import numpy as np
from scipy.optimize import golden
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
def g(alpha,ini):
    def h(alpha):
        return (obj(ini-alpha*dobj(ini)))
    return (minimize_scalar(h,method='golden').x)

alp=0
plot_path=[]
plot_path.append(init_path)
eps=1e-7
path=init_path
for i in range(0,500):
    alp=g(alp,path)
    prev=path
    path=path-alp*dobj(path)
    if(i==10 or i==20 or i==30 or i==50 or i==100 or i==200):
        plot_path.append(path)

final_path=path
plot_path.append(final_path)
plot_path.append(obstacles)
strings =['initial path' ,'path 10','path 20','path 30','path 50','path 100','path 200','final path','obstacles']
colors=['#0072B2','#E5E5E5','blue','#00FFCC','green','y','c','r','#eeefff']
for k in range(0,len(plot_path)):
    x1=np.zeros(init_path.shape[0])
    y1=np.zeros(init_path.shape[0])
    for j in range(0,init_path.shape[0],1):
        x1[j]=plot_path[k][j][0]
        y1[j]=plot_path[k][j][1]
    if k!=(len(plot_path)-1):
        plt.plot(x1,y1,label=strings[k],color=colors[k])
    else:
        plt.plot(x1,y1,"o",label=strings[k],color=colors[k])
plt.legend( loc='upper right', numpoints = 1 )

print("We can say the following things about the plot: \n" )
print("1.With the increase in number of iterations, the number of obstacles in the path are minimized and so is the distance.")
print("2.At the expense of the computational cost, the algorithm optimizes the initial path by using the shortest possible path and avoiding obstacles.")
