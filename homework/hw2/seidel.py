import numpy as np
import matplotlib.pyplot as plt
import time
import math

def accuracy(a, b):
	res = 0
	n = np.size(a)
	for i in range(n):
		res += (a[i] - b[i])**2
	return math.sqrt(res)

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

def Seidel(A, f, x):
	n = np.size(f)
	xnew = np.zeros(n)
	for i in range(n):
		s = 0
		for j in range(0, i - 1):
			s += A[i][j] * xnew[j]
		for j in range(i + 1, n):
			s += A[i][j] * x[j]
		xnew[i] = (f[i] - s) / A[i][i]
	return xnew

def diff_is_more_than_eps(a, b, eps):
	n = len(a)
	sum = 0
	for i in range(n):
		sum += abs(a[i]**2 - b[i]**2)
	if math.sqrt(sum) < eps:
		return False
	return True

def solve(A, f):
	eps = 10**(-4)
	n = np.size(f)
	xnew = np.zeros(n)
	x = xnew
	xnew = Seidel(A, f, x)
	while not diff_is_more_than_eps(x, xnew, eps) == True:
		x = xnew
		xnew = Seidel(A, f, x)
	return x

n = 100
i = 0
res_seidel = [0.0]
res_linalg = [0.0]
while True:
	A = np.random.rand(n, n)
	A = diagDomination(A)
	f = np.random.rand(n)

	x1 = np.zeros(n)
	start_time = time.time()
	x1 = solve(A, f)
	res_seidel.append(float(toFixed(time.time() - start_time, 19)))
	print(" n = %s:\n mySeidel(): %s s" % (n, res_seidel[i+1]))

	x2 = np.zeros(n)
	start_time = time.time()
	x2 = np.linalg.solve(A, f)
	res_linalg.append(float(toFixed(time.time() - start_time, 19)))
	print(" linalg.solve(): %s s" % res_linalg[i+1])
	if res_seidel[i] > 1 or res_linalg[i] > 1:
		break
	n += 100
	i += 1
	print(" accuracy ||x1 - x2|| = %s\n" % accuracy(x1, x2))

fig, ax = plt.subplots()
ax.set(facecolor = 'white',
        title = 'Сравнение с linalg, Красное - mySeidel, Синее - linalg',
        xlabel = 'Размер матрицы в сотнях',
        ylabel = 'Время выполнения')
ax.plot(res_seidel, color = 'r')
ax.plot(res_linalg, color = 'b')
plt.savefig('seidel.png', bbox_inches='tight')
plt.show()

