import numpy as np
import matplotlib.pyplot as plt

def lagrange(x, y, t):
	z = 0
	for j in range(len(y)):
		p1 = 1; p2 = 1
		for i in range(len(x)):
			if i == j:
				p1 = p1*1; p2 = p2*1
			else:
				p1 = p1*(t - x[i])
				p2 = p2*(x[j] - x[i])
		z = z + y[j]*p1 / p2
	return z

xFile = open('train.dat', 'r')
yFile = open('train.ans', 'r')
testXfile = open('test.dat', 'r')
testYfile = open('test.ans', 'w')

n = int(xFile.readline()) # fill arrays
m = int(testXfile.readline())
x = [float(i) for i in xFile.readline().split()]
y = [float(i) for i in yFile.readline().split()]

z = [float(i) for i in testXfile.readline().split()] # find new points
f = [lagrange(x, y, i) for i in z]

min_xz = min(np.min(x), np.min(z))
max_xz =  max(np.max(x), np.max(z))

xnew = np.linspace(min_xz , max_xz, 50)
ynew = [lagrange(x, y, i) for i in xnew]

xFile.close()
yFile.close()
testXfile.close()
testYfile.close()

plt.plot(x, y, 'o', xnew, ynew)
plt.plot(z, f, 'o')
plt.grid(True)
plt.show()
