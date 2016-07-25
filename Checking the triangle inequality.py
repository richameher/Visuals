import numpy as np
import matplotlib.pyplot as plt



k=np.add(x,y)
sum_1=np.sum(np.abs(k)**1)**(1)
sum_2=np.sum(np.abs(k)**2)**(1/2)
sum_5=np.sum(np.abs(k)**5)**(1/5)
sum_half=np.sum(np.abs(k)**0.5,axis=0)**(2)
norm1_x= np.sum(np.abs(x)**1,axis=0)**(1)
norm2_x= np.sum(np.abs(x)**2)**(1/2)
norm5_x= np.sum(np.abs(x)**5,axis=0)**(1/5)
normhalf_x= np.sum(np.abs(x)**0.5)**(2)

norm1_y= np.sum(np.abs(y)**1,axis=0)**(1)
norm2_y= np.sum(np.abs(y)**2)**(1/2)
norm5_y= np.sum(np.abs(y)**5)**(1/5)
normhalf_y= np.sum(np.abs(y)**0.5)**(2)

norm1_sum=norm1_x+norm1_y
norm2_sum=norm2_x+norm2_y
norm5_sum=norm5_x+norm5_y
normhalf_sum=normhalf_x+normhalf_y

norm_sum=np.array([norm1_sum, norm2_sum, norm5_sum,normhalf_sum])

sum_norm=np.array([sum_1,sum_2,sum_5,sum_half])

alpha = np.linspace(0, 2*np.pi, 500, endpoint=True)
a = np.cos(alpha)
b = np.sin(alpha)
vecs = np.array([a,b])



norms_one = np.sum(np.abs(vecs)**1,axis=0)**(1)
norm_vecs_one = vecs/norms_one

norms_two = np.sum(np.abs(vecs)**2,axis=0)**(1/2)
norm_vecs_two = vecs/norms_two

norms_five = np.sum(np.abs(vecs)**5,axis=0)**(1/5)
norm_vecs_five = vecs/norms_five


norms_half = np.sum(np.abs(vecs)**0.5,axis=0)**(2)
norm_vecs = vecs/norms_half

plt.figure(1)
plt.title("p=0.5")
norm_vecshalf=normhalf_sum*norm_vecs
norm_vecsxhalf=normhalf_x*norm_vecs
norm_vecsyhalf=normhalf_y*norm_vecs
norm_vecsumhalf=sum_half*norm_vecs
plt.grid()
plt.gca().set_aspect("equal")
plt.plot(norm_vecshalf[0], norm_vecshalf[1],'b')
plt.plot(norm_vecsxhalf[0], norm_vecsxhalf[1],'g')
plt.plot(norm_vecsyhalf[0]+x[0], norm_vecsyhalf[1]+x[1],'r')
plt.plot(norm_vecsumhalf[0], norm_vecsumhalf[1],'c')
ax = plt.axes()
ax.arrow(0, 0,x[0],x[1], head_width=0.1, head_length=0.2,length_includes_head=True)
ax.arrow(0, 0,k[0],k[1], head_width=0.1, head_length=0.2,length_includes_head=True)
ax.arrow(x[0],x[1],y[0],y[1], head_width=0.1, head_length=0.2,length_includes_head=True)
plt.show()
plt.xlim([-3, 3])
plt.ylim([-3, 3])

plt.figure(2)
plt.title("NORM-1")
norm_vecsone=norm1_sum*norm_vecs_one
norm_vecsxone=norm1_x*norm_vecs_one
norm_vecsyone=norm1_y*norm_vecs_one
norm_vecsumone=sum_1*norm_vecs_one
plt.grid()
plt.gca().set_aspect("equal")
plt.plot(norm_vecsone[0], norm_vecsone[1],'b')
plt.plot(norm_vecsxone[0], norm_vecsxone[1],'g')
plt.plot(norm_vecsyone[0]+x[0], norm_vecsyone[1]+x[1],'r')
plt.plot(norm_vecsumone[0], norm_vecsumone[1],'c')
ax = plt.axes()
ax.arrow(0, 0,x[0],x[1], head_width=0.1, head_length=0.2,length_includes_head=True)
ax.arrow(0, 0,k[0],k[1], head_width=0.1, head_length=0.2,length_includes_head=True)
ax.arrow(x[0],x[1],y[0],y[1], head_width=0.1, head_length=0.2,length_includes_head=True)
plt.show()
plt.xlim([-2, 2])
plt.ylim([-2, 2])

plt.figure(3)
plt.title("NORM-2")
norm_vec2=norm2_sum*norm_vecs_two
norm_vecsx2=norm2_x*norm_vecs_two
norm_vecsy2=norm2_y*norm_vecs_two
norm_vecsum2=sum_2*norm_vecs_two
plt.grid()
plt.gca().set_aspect("equal")
plt.plot(norm_vec2[0], norm_vec2[1],'b')
plt.plot(norm_vecsx2[0], norm_vecsx2[1],'g')
plt.plot(norm_vecsy2[0]+x[0], norm_vecsy2[1]+x[1],'r')
plt.plot(norm_vecsum2[0], norm_vecsum2[1],'c')
ax = plt.axes()
ax.arrow(0, 0,x[0],x[1], head_width=0.1, head_length=0.2,length_includes_head=True)
ax.arrow(0, 0,k[0],k[1], head_width=0.1, head_length=0.2,length_includes_head=True)
ax.arrow(x[0],x[1],y[0],y[1], head_width=0.1, head_length=0.2,length_includes_head=True)
plt.show()
plt.xlim([-2, 2])
plt.ylim([-2, 2])

plt.figure(4)
plt.title("NORM-5")
norm_vec5=norm5_sum*norm_vecs_five
norm_vecsx5=norm5_x*norm_vecs_five
norm_vecsy5=norm5_y*norm_vecs_five
norm_vecsum5=sum_5*norm_vecs_five
plt.grid()
plt.gca().set_aspect("equal")
plt.plot(norm_vec5[0], norm_vec5[1],'b')
plt.plot(norm_vecsx5[0], norm_vecsx5[1],'g')
plt.plot(norm_vecsy5[0]+x[0], norm_vecsy5[1]+x[1],'r')
plt.plot(norm_vecsum5[0], norm_vecsum5[1],'c')
ax = plt.axes()
ax.arrow(0, 0,x[0],x[1], head_width=0.1, head_length=0.2,length_includes_head=True)
ax.arrow(0, 0,k[0],k[1], head_width=0.1, head_length=0.2,length_includes_head=True)
ax.arrow(x[0],x[1],y[0],y[1], head_width=0.1, head_length=0.2,length_includes_head=True)
plt.show()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
print("x:",x,"y:",y)
print("NOTE : BLUE:||x||+||y||, GREEN:||x||, RED:||y||,CYAN:||x+y||")
print("According to triange inequality ||x||+||y||>= ||x+y||.\n ")
print("As the blue ball (||x||+||y||) is always bigger or equal to Cyan ball(||x+y||),")
print("we can say that the triangle equality holds for norms,1,2,5.\n")
print("However the triangle inequality does not apply for p=0.5 as sometimes Cyan ball is bigger than Blue ball,ie;||x+y|| > ||x||+||y||")
print("Consider vectors x=[1,0] and y=[0,1], the triangle inequality for p=0.5 is unsatisfied ie;(||x+y||>||x||+||y||) but is satisfied \n in case of p>0.")
