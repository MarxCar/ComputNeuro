import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(x, y, z, s, r, b):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01
stepCnt = 5000


def attractor(param=[10,28,2.667]):
	# Need one more for the initial values
	s, r, b = param
	xs = np.empty((stepCnt + 1,))
	ys = np.empty((stepCnt + 1,))
	zs = np.empty((stepCnt + 1,))

	# Setting initial values
	xs[0], ys[0], zs[0] = (0., 1., 1.05)

	# Stepping through "time".
	for i in range(stepCnt):
	    # Derivatives of the X, Y, Z state
	    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
	    xs[i + 1] = xs[i] + (x_dot * dt)
	    ys[i + 1] = ys[i] + (y_dot * dt)
	    zs[i + 1] = zs[i] + (z_dot * dt)
	return xs, ys


num_graphs = 10

s_range = [5,20]
r_range = [10,50]
b_range = [1,3]
parameters = ['s', 'r', 'b']
param_ranges = [s_range, r_range, b_range]
count = 0	
for param_range in param_ranges:
	for index, param_val in zip(range(1 + count*num_graphs, (count+2)*num_graphs),np.linspace(param_range[0],param_range[1],num_graphs)):

		param_array = [10,28,2.667]
		param_array[count] = param_val
		plt.subplot(3,num_graphs,index)

		plt.title("%s=%.2f" % (parameters[count],param_val))
		plt.axis('off')
		xs, ys = attractor(param_array)
		plt.plot(xs,ys)
	
	count += 1

plt.savefig("X&Y.png")

#plt.set_title("Lorenz Attractor")
plt.show()
