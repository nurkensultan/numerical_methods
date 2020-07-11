import numpy as np
import math
import time
import matplotlib.pyplot as plt

def accuracy(a, b):
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

def Cholesky(A, f):
	n = len(f)
	L = np.zeros((n, n))
	for i in range(n):
		for k in range(i + 1):
			tmp = sum(L[i][j] * L[k][j] for j in range(k))
			if i == k:
				L[i][k] = math.sqrt(A[i][i] - tmp)
			else:
				L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp))
	U = np.transpose(L)
	y = np.zeros(n)
	x = np.zeros(n)
	for i in range(n - 1, -1, -1):
		y[i] = f[i]
		for j in range(i + 1, n):
			y[i] -= L[i][j] * y[j]
	for i in range(n - 1, -1, -1):
		x[i] = y[i]
		for j in range(i + 1, n):
			x[i] -= U[i][j] * x[j]
	return x

n = 100
i = 0
res_cholesky = [0.0]
res_linalg = [0.0]
while True:
	B = np.random.rand(n, n)
	B = diagDomination(B)
	Bt = B.transpose()
	A = np.dot(Bt, B)
	f = np.random.rand(n)
	x = np.zeros(n)

	start_time = time.time()
	x = Cholesky(A, f)
	res_cholesky.append(float(toFixed(time.time() - start_time, 19)))

	start_time = time.time()
	x = np.zeros(n)
	y = np.zeros(n)
	L1 = np.linalg.cholesky(A)
	U1 = np.transpose(L1)
	y = np.linalg.solve(L1, f)
	x = np.linalg.solve(U1, y)
	res_linalg.append(float(toFixed(time.time() - start_time, 19)))
	n += 100
	if res_cholesky[i] > 1 or res_linalg[i] > 1:
		break
	i += 1

fig, ax = plt.subplots()
ax.set(facecolor = 'white',
	title = 'Сравнение с linalg, Красное - myCholesky, Синее - linalg',
	xlabel = 'Размер матрицы в сотнях',
	ylabel = 'Время выполнения')
ax.plot(res_cholesky, color = 'r')
ax.plot(res_linalg, color = 'b')
plt.savefig('cholesky.png', bbox_inches='tight')
plt.show()
