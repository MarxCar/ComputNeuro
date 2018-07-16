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


def attractor(s=10,r=28,b=2.667):
	# Need one more for the initial values
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


#ax = fig.gca(projection='3d')
'''
plt.subplot(121)
plt.title("b=2.667")
plt.plot(x_vals, y_vals#, zs, lw=0.5
)
plt.subplot(122)
plt.title("b=2")
plt.plot(attractor(b=2)[0], attractor(b=2)[1])
'''
num_graphs = 3

s_range = [5,20]
r_range = [10,50]
b_range = [1,3]
param_ranges = [s_range, r_range, b_range]
count = 1	
for param_range in param_ranges:
	for index, param_val in zip(range(1 + (count*3),num_graphs+1),np.linspace(param_range[0],param_range[1],num_graphs)):
		print(str(num_graphs) + "" + str(len(param_range)) + "" + str(index))
		plt.subplot(num_graphs,len(param_range),index)

		plt.title("b=%f" % (b_val))
		xs, ys = attractor(b=b_val)
		plt.plot(xs,ys)
		print(str(num_graphs) + "3" + str(index))
	count += 1



"""

for index, b_val in zip(range(1,num_graphs+1),np.linspace(b_range[0],b_range[1],num_graphs)):
	plt.subplot(num_graphs,3,index)
	plt.title("b=%f" % (b_val))
	xs, ys = attractor(b=b_val)
	plt.plot(xs,ys)
	print(str(num_graphs) + "3" + str(index))

for index, s_val in zip(range(num_graphs+1,(num_graphs*2)+1),np.linspace(s_range[0],s_range[1],num_graphs)):
	plt.subplot(num_graphs,3,index)
	print(str(num_graphs) + "3" + str(index))
	plt.title("s=%f" % (s_val))
	xs, ys = attractor(s=s_val)
	plt.plot(xs,ys)
for index, r_val in zip(range(num_graphs+4,(num_graphs*3)+1),np.linspace(r_range[0],r_range[1],num_graphs)):
	plt.subplot(num_graphs,3,index)
	print(str(num_graphs) + "3" + str(index))
	plt.title("r=%f" % (r_val))
	xs, ys = attractor(r=r_val)
	plt.plot(xs,ys)

"""
#plt.savefig("thirdshit.png")
#plt.set_xlabel("X Axis")
#plt.set_ylabel("Y Axis")
#ax.set_zlabel("Z Axis")
#plt.set_title("Lorenz Attractor")

plt.show()
