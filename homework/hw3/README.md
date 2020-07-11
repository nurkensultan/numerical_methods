3.1 Linear interpolation on an uneven grid
=====================================================
file: linear.py

3.2 Lagrange interpolation on an uneven grid
=====================================================
file: lagrange.py

3.3 Spline interpolation on an uniform grid
=====================================================
file: spline.py

Details
=====================================================
Input data:
1) train.dat - old grid
2) train.ans - function values on the old grid
3) test.dat - new grid

The task is to build 3 interpolation models and apply them on a new grid.

Output data:
1) test.ans - function values on the new grid

Programs generate and solve linear equations (n*n sized) until its execution time reaches 1 second.
Also they compare solutions with numpy methods, build graphs of code execution time. 

How to launch code: $ python3 <name>.py
