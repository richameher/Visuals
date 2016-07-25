import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt

def random_gen_crossed_circles(height, width):
    i, j = np.mgrid[:height,:width]
    radius = int(height * 0.35)
    center_i = int(height / 2)

    half_center_dist = radius - 2
    center_j_1 = int(width / 2) - half_center_dist
    center_j_2 = int(width / 2) + half_center_dist

    is_in_domain = (
                    ((i-center_i)**2 + (j-center_j_1)**2 < radius**2)
                    | ((i-center_i)**2 + (j-center_j_2)**2 < radius**2)
                    )
    return is_in_domain

def random_gen_o_square(height, width):
    i, j = np.mgrid[:height,:width]
    radius = int(width / 5)
    center = int(width / 2)
    center_dist_from_board = int(1.5 * radius)

    left_up_i = 0 + center_dist_from_board
    left_up_j = 0 + center_dist_from_board

    right_up_i = 0 + center_dist_from_board
    right_up_j = width - center_dist_from_board

    left_bottom_i = height - center_dist_from_board
    left_bottom_j = 0 + center_dist_from_board

    right_bottom_i = height - center_dist_from_board
    right_bottom_j = width - center_dist_from_board

    p = 4
    is_in_domain = (
                    ((i-center)**p + (j-center)**p < radius**p)
                    | ((i-left_up_i)**p + (j-left_up_j)**p < radius**p)
                    | ((i-right_up_i)**p + (j-right_up_j)**p < radius**p)
                    | ((i-left_bottom_i)**p + (j-left_bottom_j)**p < radius**p)
                    | ((i-right_bottom_i)**p + (j-right_bottom_j)**p < radius**p)
                    )

    return is_in_domain

def plot_eigenvector(v):
    if nvariables != v.shape[0]:
        print ("Fail to plot: shape of vector not correct.")
        return

    plot_array = np.zeros((height, width))
    plot_array[is_in_domain_ref] = 1/(1e-2+np.abs(v))
    pt.imshow(plot_array, cmap="gray")

# Generating domain
if np.random.randint(2) == 0:
    width = 60 + np.random.randint(-5, 6)
    height = 30 + np.random.randint(-3, 4)
    is_in_domain_ref = random_gen_crossed_circles(height, width)
else:
    height = width = 40 + np.random.randint(-5, 6)
    is_in_domain_ref = random_gen_o_square(height, width)

index_to_i_ref, index_to_j_ref = np.where(is_in_domain_ref)
nvariables_ref = len(index_to_i_ref)

ij_to_index_ref = np.zeros((height, width), dtype=np.int32)
ij_to_index_ref[is_in_domain_ref] = np.arange(nvariables_ref)

is_in_domain = is_in_domain_ref.copy()
index_to_i = index_to_i_ref.copy()
index_to_j = index_to_j_ref.copy()
nvariables = nvariables_ref
ij_to_index = ij_to_index_ref.copy()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import numpy.linalg as la
W=np.zeros(nvariables*nvariables)
bound_col=ij_to_index.shape[1]-1
W.shape=(nvariables,nvariables)
img=ij_to_index[:,:]
plt.imshow(img)
plt.title('Domain')
bound=ij_to_index.shape[0]-1
#print(index_to_j.shape,index_to_i.shape,height,width,is_in_domain.shape,is_in_domain,nvariables,ij_to_index.shape)
np.set_printoptions(threshold=np.nan)
#print(ij_to_index.shape,nvariables)

for i in range(0,ij_to_index.shape[0],1):
    for j in range(0,ij_to_index.shape[1],1):
            if is_in_domain[i][j]!=False:
                    W[ij_to_index[i][j]][ij_to_index[i][j]]=4
                    if i<bound:
                        if is_in_domain[i+1][j]!=False :
                            W[ij_to_index[i][j]][ij_to_index[i+1][j]]=-1
                    if i>0:
                        if is_in_domain[i-1][j]!=0:
                            W[ij_to_index[i][j]][ij_to_index[i-1][j]]=-1
                    if j<bound_col:
                        if is_in_domain[i][j+1]!=0:
                            W[ij_to_index[i][j]][ij_to_index[i][j+1]]=-1
                    if j>0:
                        if is_in_domain[i][j-1]!=0:
                            W[ij_to_index[i][j]][ij_to_index[i][j-1]]=-1
vect,vect1=la.eigh(W)
u_12=vect1[:,11]

plt.figure(2)
plt.title('W')
plt.spy(W,markersize=0.1)
plt.figure(3)
plt.title('Eigenvector')
plot_eigenvector(u_12)
