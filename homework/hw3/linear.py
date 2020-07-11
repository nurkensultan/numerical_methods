import numpy as np
import scipy.linalg as sla
import time
import matplotlib.pyplot as plt

xFile = open('train.dat', 'r')
yFile = open('train.ans', 'r')
testXfile = open('test.dat', 'r')
testYfile = open('test.ans', 'w')

n = int(xFile.readline()) # fill arrays
m = int(testXfile.readline())
x = [float(i) for i in xFile.readline().split()]
y = [float(i) for i in yFile.readline().split()]
z = [float(i) for i in testXfile.readline().split()]

a = np.zeros(n) # coefs
b = np.zeros(n)
for i in range(n-1):
	a[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
	b[i] = y[i]

f = np.zeros(m) # my approaching
for j in range(m):
	for i in range(0,n-1):
		if (z[j] < x[0]): # < [x_0); x_{n-1}]
			f[j] = a[0] * (z[j] - x[0]) + b[0]
		if (z[j] >= x[n - 1]): # > [x_0; x_{n-1}]
			f[j] = a[n - 2] * (z[j] - x[n - 2]) + b[n - 2]
		if (x[i] < z[j] and z[j] <= x[i+1]): # in [x_i; x_{i + 1})
			f[j] = a[i] * (z[j] - x[i]) + b[i]
	testYfile.write(str(f[j]) + ' ')

xFile.close()
yFile.close()
testXfile.close()
testYfile.close()

plt.plot(x, y)
plt.plot(x, y, 'o')
plt.plot(z, f, 'o')
plt.plot(z, f)
plt.show()
