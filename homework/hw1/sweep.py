import numpy as np
import scipy.linalg as sl
import time
import matplotlib.pyplot as plt

def sweep(a, b, c, f, n):
    alpha = np.array([0.0] * (n + 1))
    beta = np.array([0.0] * (n + 1))
    for i in range(n):
        alpha[i + 1] = -c[i] / (a[i] * alpha[i] + b[i])
        beta[i + 1] = (f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + b[i])
    x = np.array([0.0] * n)
    x[n - 1] = beta[n]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]
    return x

def matrix(a, b, c, n):
    a_ = np.array([0] + c[:-1].tolist())
    b_ = b
    c_ = np.array(a[1:].tolist() + [0])
    A = np.array([a_, b_, c_])
    return A

def fixed(num, digits):
    return f"{num:.{digits}f}"

i = 0
n = 1000
a = np.zeros(n)
b = np.zeros(n)
c = np.zeros(n)
f = np.zeros(n)

res_sweep = np.zeros(n)
res_linalg = np.zeros(n)
time_sweep = [0.0]
time_linalg = [0.0]

while n <= 200000:
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.random.rand(n)
    f = np.random.rand(n)
    A = matrix(a, b, c, n)

    start_time = time.time()
    res_sweep = sweep(a, b, c, f, n)
    time_sweep.append(float(fixed(time.time() - start_time, 10)))

    start_time = time.time()
    res_linalg = sl.solve_banded((1, 1), A, f)
    time_linalg.append(float(fixed(time.time() - start_time, 10)))

    i += 1
    n += 1000
    if time_sweep[i] > 1:
        break

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(title = 'Сравнение с linalg, красное - linalg, синее - sweep()',
   xlabel = 'Размер матрицы в 1000',
   ylabel = 'Время выполнения в секундах')
ax.plot(time_sweep, color = 'b')
ax.plot(time_linalg, color = 'r')
plt.savefig('sweep.png', bbox_inches='tight')
plt.show()
