import numpy as np
import matplotlib.pyplot as plt
import time
from math import *

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

def Gauss(A, f):
	n = np.size(f)
	for k in range(n):
		f[k] /= A[k][k]
		A[k] /= A[k][k]
		for i in range(k + 1, n):
			f[i] -= f[k] * A[i][k]
			A[i] -= A[k] * A[i][k]
			A[i][k] = 0
	x = np.zeros(n)
	for i in range(n - 1, -1, -1):
		x[i] = f[i]
		for j in range(i + 1, n):
			x[i] -= A[i][j] * x[j]
	return x

n = 100
i = 0
res_gauss = [0.0]
res_linalg = [0.0]
while True:
# подготовим матрицу А с диаг. преобладанием, векторы f, x
	A = np.random.rand(n, n)
	A = diagDomination(A)
	f = np.random.rand(n)

	x1 = np.zeros(n)
	start_time = time.time()
	x1 = Gauss(A, f)
	res_gauss.append(float(toFixed(time.time() - start_time, 19)))

	x2 = np.zeros(n)
	start_time = time.time()
	x2 = np.linalg.solve(A, f)
	res_linalg.append(float(toFixed(time.time() - start_time, 19)))

	n += 100
	if res_gauss[i] > 1 or res_linalg[i] > 1:
		break
	i += 1

fig, ax = plt.subplots()
ax.set(facecolor = 'yellow',
	title = 'Сравнение с linalg. Красное - myGauss, синее - linalg',
	xlabel = 'Размер матрицы в сотнях',
	ylabel = 'Время выполнения в секундах')
ax.plot(res_gauss, color = 'r')
ax.plot(res_linalg, color = 'b')
plt.savefig('gauss.png', bbox_inches='tight')
plt.show()
