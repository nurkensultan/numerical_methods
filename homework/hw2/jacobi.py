import numpy as np
import matplotlib.pyplot as plt
import time
from math import *

def diff(a, b):
	res = 0
	n = np.size(a)
	for i in range(n):
		res += (a[i] - b[i])**2
	return sqrt(res)

def toFixed(num, digits=0):
	return f"{num:.{digits}f}"

def diagDomination(A):
	n = np.size(A[0])
	for i in range(n):
		jmax = 0
		max = 0
		for j in range(n):
			if abs(A[i][j]) > abs(max):
				max = A[i][j]
				jmax = j
		if A[i][i] != A[i][jmax]:
			A[i][i], A[i][jmax] = A[i][jmax], A[i][i]
	return A

def Jacobi(A, f, x):
	n = np.size(f)
	xnew = np.zeros(n)
	n = np.size(f)
	for i in range(n):
		s = 0
		for j in range(i):
			s += A[i][j] * x[j]
		for j in range(i + 1, n):
			s += A[i][j] * x[j]
		xnew[i] = (f[i] - s) / A[i][i]
	return xnew

def solve(A, f):
	n = np.size(f)
	xnew = np.zeros(n)
	while True:
		x = np.array(xnew)
		xnew = Jacobi(A, f, x)
		if diff(x, xnew) < eps:
			break
	return xnew

eps = 0.1
n = 100
i = 0
res_jacobi = [0.0]
res_linalg = [0.0]
while True:
	A = np.random.rand(n, n)
	A = diagDomination(A)
	f = np.random.rand(n)
	x1 = np.zeros(n)

	start_time = time.time()
	x1 = Jacobi(A, f, x1)
	res_jacobi.append(float(toFixed(time.time() - start_time, 19)))
	print("n = %s:\n myJacobi(): %s s" % (n, res_jacobi[i+1]))

	x2 = np.zeros(n)
	start_time = time.time()
	x2 = np.linalg.solve(A, f)
	res_linalg.append(float(toFixed(time.time() - start_time, 19)))
	print(" linalg.solve(): %s s" % res_linalg[i+1])
	n += 100
	if res_jacobi[i+1] > 1 or res_linalg[i+1] > 1:
		break
	i += 1
	print(" accuracy ||x1 - x2|| = %s\n" % diff(x1, x2))

fig, ax = plt.subplots()
ax.set(facecolor = 'white',
	title = 'Сравнение с linalg, Красное - myJacobi, Синее - linalg',
	xlabel = 'Размер матрицы в сотнях',
	ylabel = 'Время выполнения')
ax.plot(res_jacobi, color = 'r')
ax.plot(res_linalg, color = 'b')
plt.savefig('jacobi.png', bbox_inches='tight')
plt.show()

