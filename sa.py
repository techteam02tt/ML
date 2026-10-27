1.Gauss elimination


def GEM(n, A, b):
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - m * A[k][j]
            b[i] = b[i] - m * b[k]
            A[i][k] = 0.0
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        s = sum(A[k][j] * x[j] for j in range(k + 1, n))
        x[k] = (b[k] - s) / A[k][k]
    return x

n = int(input("Enter the order of the matrix: "))
print("Enter the coefficient matrix A:")
A = [list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)]
b = list(map(float, input("Enter the RHS vector b: ").split()))

x = GEM(n, A, b)
print("\nSolution vector:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")


OUTPUT

Enter the order of the matrix: 3
Enter the coefficient matrix A:
Row 1: 2 1 -1
Row 2: -3 -1 2
Row 3: -2 1 2
Enter the RHS vector b: 8 -11 -3

Solution vector:
x1 = 2.000000
x2 = 3.000000
x3 = -1.000000


2.GAUSS ELI AND ROW PIVOTING


def GEM(n, A, b, tol=1e-2):
    for k in range(n - 1):
        if abs(A[k][k]) <= tol:
            p = max(range(k, n), key=lambda q: abs(A[q][k]))
            if abs(A[p][k]) <= tol:
                raise ValueError("Singular matrix")
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
            print(f"Pivot is zero, rows {k+1} and {p+1} swapped")
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - m * A[k][j]
            b[i] = b[i] - m * b[k]
            A[i][k] = 0.0
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        s = sum(A[k][j] * x[j] for j in range(k + 1, n))
        x[k] = (b[k] - s) / A[k][k]
    return x

n = int(input("Enter the order of the matrix: "))
print("Enter the coefficient matrix A:")
A = [list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)]
b = list(map(float, input("Enter the RHS vector b: ").split()))

x = GEM(n, A, b)
print("\nSolution vector:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")


OUTPUT

Enter the order of the matrix: 3
Enter the coefficient matrix A:
Row 1: 0 1 -1
Row 2: -3 -1 2
Row 3: -2 1 2
Enter the RHS vector b: 4 -11 -3
Pivot is zero, rows 1 and 2 swapped

Solution vector:
x1 = 2.000000
x2 = 3.000000
x3 = -1.000000


3.JACOBI


def gauss_jacobi(A, b, x_old, TOL, KMAX):
    n = len(b)
    for k in range(1, KMAX + 1):
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        error = max(abs(x_new[i] - x_old[i]) for i in range(n))
        print(f"Iteration {k}: {[round(v, 6) for v in x_new]}  Error = {error:.6f}")
        if error <= TOL:
            print(f"\nConverged after {k} iterations")
            return x_new
        x_old = x_new
    print("\nMaximum iterations reached")
    return x_old

n = int(input("Enter the order of the matrix: "))
print("Enter the coefficient matrix A:")
A = [list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)]
b = list(map(float, input("Enter the RHS vector b: ").split()))
x0 = list(map(float, input("Enter the initial guess x0: ").split()))
TOL = float(input("Enter tolerance TOL: "))
KMAX = int(input("Enter maximum iterations KMAX: "))

x = gauss_jacobi(A, b, x0, TOL, KMAX)
print("Solution:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")


OUTPUT

Enter the order of the matrix: 3
Enter the coefficient matrix A:
Row 1: 10 1 1
Row 2: 2 10 1
Row 3: 2 2 10
Enter the RHS vector b: 12 13 14
Enter the initial guess x0: 0 0 0
Enter tolerance TOL: 0.001
Enter maximum iterations KMAX: 20
Iteration 1: [1.2, 1.3, 1.4]  Error = 1.400000
Iteration 2: [0.93, 0.92, 0.9]  Error = 0.500000
Iteration 3: [1.018, 1.024, 1.03]  Error = 0.130000
Iteration 4: [0.9946, 0.9934, 0.9916]  Error = 0.038400
Iteration 5: [1.0015, 1.00192, 1.0024]  Error = 0.010800
Iteration 6: [0.999568, 0.99946, 0.999316]  Error = 0.003084
Iteration 7: [1.000122, 1.000155, 1.000194]  Error = 0.000878

Converged after 7 iterations
Solution:
x1 = 1.000122
x2 = 1.000155
x3 = 1.000194


4.SEIDAL


def gauss_seidel(A, b, x_old, TOL, KMAX):
    n = len(b)
    for k in range(1, KMAX + 1):
        x_new = [0.0] * n
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        error = max(abs(x_new[i] - x_old[i]) for i in range(n))
        print(f"Iteration {k}: {[round(v, 6) for v in x_new]}  Error = {error:.6f}")
        if error <= TOL:
            print(f"\nConverged after {k} iterations")
            return x_new
        x_old = x_new
    print("\nMaximum iterations reached")
    return x_old

n = int(input("Enter the order of the matrix: "))
print("Enter the coefficient matrix A:")
A = [list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)]
b = list(map(float, input("Enter the RHS vector b: ").split()))
x0 = list(map(float, input("Enter the initial guess x0: ").split()))
TOL = float(input("Enter tolerance TOL: "))
KMAX = int(input("Enter maximum iterations KMAX: "))

x = gauss_seidel(A, b, x0, TOL, KMAX)
print("Solution:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")


OUTPUT

Enter the order of the matrix: 3
Enter the coefficient matrix A:
Row 1: 10 1 1
Row 2: 2 10 1
Row 3: 2 2 10
Enter the RHS vector b: 12 13 14
Enter the initial guess x0: 0 0 0
Enter tolerance TOL: 0.001
Enter maximum iterations KMAX: 20
Iteration 1: [1.2, 1.06, 0.948]  Error = 1.200000
Iteration 2: [0.9992, 1.00536, 0.999088]  Error = 0.200800
Iteration 3: [0.999555, 1.00018, 1.000053]  Error = 0.005180
Iteration 4: [0.999977, 0.999999, 1.000005]  Error = 0.000421

Converged after 4 iterations
Solution:
x1 = 0.999977
x2 = 0.999999
x3 = 1.000005


5.POWER METHOD


import math

n = int(input("Enter order of matrix: "))
print("Enter matrix A:")
A = [list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)]
v = list(map(float, input("Enter initial vector: ").split()))
TOL = float(input("Enter tolerance: "))
KMAX = int(input("Enter maximum iterations: "))

lam = 0
for k in range(1, KMAX + 1):
    w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
    c = max(w, key=abs)
    v_new = [wi / c for wi in w]
    error = max(abs(lam - c), math.dist(v, v_new))
    lam, v = c, v_new
    print(f"Iteration {k}: Eigen Value = {lam:.6f}, Eigen Vector = {[round(x, 6) for x in v]}")
    if error <= TOL:
        break

print(f"\nDominant Eigen Value = {lam:.6f}")
print(f"Dominant Eigen Vector = {[round(x, 6) for x in v]}")


OUTPUT

Enter order of matrix: 2
Enter matrix A:
Row 1: 4 1
Row 2: 2 3
Enter initial vector: 1 1
Enter tolerance: 0.001
Enter maximum iterations: 20
Iteration 1: Eigen Value = 5.000000, Eigen Vector = [1.0, 1.0]
Iteration 2: Eigen Value = 5.000000, Eigen Vector = [1.0, 1.0]

Dominant Eigen Value = 5.000000
Dominant Eigen Vector = [1.0, 1.0]


6.JORDAN INVERSE


n = int(input("Enter the order of the matrix: "))
print("Enter matrix A:")
A = [list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)]

aug = [A[i] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
for k in range(n):
    if abs(aug[k][k]) < 1e-12:
        for p in range(k + 1, n):
            if abs(aug[p][k]) > 1e-12:
                aug[k], aug[p] = aug[p], aug[k]
                break
        else:
            exit("Matrix is singular. Inverse does not exist")
    pivot = aug[k][k]
    aug[k] = [v / pivot for v in aug[k]]
    for i in range(n):
        if i != k:
            factor = aug[i][k]
            aug[i] = [aug[i][j] - factor * aug[k][j] for j in range(2 * n)]

print("\nInverse matrix:")
for row in aug:
    print("  ".join(f"{v:9.4f}" for v in row[n:]))


OUTPUT

Enter the order of the matrix: 3
Enter matrix A:
Row 1: 1 2 3
Row 2: 0 1 4
Row 3: 5 6 0

Inverse matrix:
 -24.0000    18.0000     5.0000
  20.0000   -15.0000    -4.0000
  -5.0000     4.0000     1.0000


7.LAGRANGE


import sympy as sp

n = int(input("Enter the number of data points: "))
print("Enter x and f(x) on each line:")
pts = [list(map(float, input().split())) for _ in range(n)]
xt = float(input("Enter the value of x to interpolate: "))

X = sp.Symbol('x')
P, value, ops, terms = 0, 0.0, 0, []
for i in range(n):
    L, p = 1, 1.0
    for j in range(n):
        if j != i:
            L *= (X - pts[j][0]) / (pts[i][0] - pts[j][0])
            p *= (xt - pts[j][0]) / (pts[i][0] - pts[j][0])
            ops += 4
    P += L * pts[i][1]
    value += p * pts[i][1]
    ops += 1
    num = "".join(f"(x-{pts[j][0]:g})" for j in range(n) if j != i)
    den = "".join(f"({pts[i][0]:g}-{pts[j][0]:g})" for j in range(n) if j != i)
    terms.append(f"{pts[i][1]:g}{num}/{den}")
ops += n - 1

poly = sp.nsimplify(sp.expand(P))
c = sp.Poly(poly, X).all_coeffs()
r = float(c[0])
for a in c[1:]:
    r = r * xt + float(a)

print("\n(a) Lagrange form: P(x) =", " + ".join(terms))
print("(b) Standard form: P(x) =", poly)
print(f"(c) P({xt:g}) using Lagrange form = {value:.6f}, operations = {ops}")
print("(d) Operations to convert to standard form =", sp.count_ops(poly))
print(f"(e) P({xt:g}) using Horner's method = {r:.6f}, operations = {2 * (len(c) - 1)}")


OUTPUT

Enter the number of data points: 3
Enter x and f(x) on each line:
1 1
2 4
3 9
Enter the value of x to interpolate: 2.5

(a) Lagrange form: P(x) = 1(x-2)(x-3)/(1-2)(1-3) + 4(x-1)(x-3)/(2-1)(2-3) + 9(x-1)(x-2)/(3-1)(3-2)
(b) Standard form: P(x) = x**2
(c) P(2.5) using Lagrange form = 6.250000, operations = 29
(d) Operations to convert to standard form = 1
(e) P(2.5) using Horner's method = 6.250000, operations = 4


8.NEWTON DIVIDED DIFF


n = int(input("Enter the number of data points: "))
print("Enter x and y on each line:")
pts = [list(map(float, input().split())) for _ in range(n)]
x = [p[0] for p in pts]
y = [p[1] for p in pts]

d = [[0.0] * n for _ in range(n)]
d[0] = y
for i in range(1, n):
    for j in range(n - i):
        d[i][j] = (d[i-1][j+1] - d[i-1][j]) / (x[i+j] - x[j])

print("\nDivided difference table:")
for j in range(n):
    print(f"{x[j]:8.4f}" + "".join(f"{d[i][j]:12.4f}" for i in range(n - j)))

l = int(input("\nEnter the number of values to interpolate: "))
for _ in range(l):
    X = float(input("Enter x: "))
    Y, p = d[0][0], 1.0
    for i in range(n - 1):
        p = p * (X - x[i])
        Y = Y + p * d[i+1][0]
    print(f"P({X:g}) = {Y:.6f}")


OUTPUT

Enter the number of data points: 3
Enter x and y on each line:
1 1
2 4
3 9

Divided difference table:
  1.0000      1.0000      3.0000      1.0000
  2.0000      4.0000      5.0000
  3.0000      9.0000

Enter the number of values to interpolate: 1
Enter x: 2.5
P(2.5) = 6.250000


9.FORWARD DIFF


x0 = float(input("Enter the first value of x (x0): "))
h = float(input("Enter the step length h: "))
n = int(input("Enter the number of data points: "))
y = list(map(float, input(f"Enter the {n} values of y: ").split()))

d = [[0.0] * n for _ in range(n)]
d[0] = y
for i in range(1, n):
    for j in range(n - i):
        d[i][j] = d[i-1][j+1] - d[i-1][j]

print("\nForward difference table:")
for j in range(n):
    print(f"{x0 + j*h:8.4f}" + "".join(f"{d[i][j]:12.4f}" for i in range(n - j)))

l = int(input("\nEnter the number of values to interpolate: "))
for _ in range(l):
    X = float(input("Enter x: "))
    u = (X - x0) / h
    Y, p = d[0][0], 1.0
    for i in range(1, n):
        p = p * (u - i + 1) / i
        Y = Y + p * d[i][0]
    print(f"u = {u:.4f}, P({X:g}) = {Y:.6f}")


OUTPUT

Enter the first value of x (x0): 1
Enter the step length h: 1
Enter the number of data points: 3
Enter the 3 values of y: 1 4 9

Forward difference table:
  1.0000      1.0000      3.0000      2.0000
  2.0000      4.0000      5.0000
  3.0000      9.0000

Enter the number of values to interpolate: 1
Enter x: 2.5
u = 1.5000, P(2.5) = 6.250000


10.BACKWARD DIFF


x0 = float(input("Enter the first value of x (x0): "))
h = float(input("Enter the step length h: "))
n = int(input("Enter the number of data points: "))
y = list(map(float, input(f"Enter the {n} values of y: ").split()))

d = [[0.0] * n for _ in range(n)]
d[0] = y
for i in range(1, n):
    for j in range(n - i):
        d[i][j] = d[i-1][j+1] - d[i-1][j]

print("\nBackward difference table:")
for j in range(n):
    print(f"{x0 + j*h:8.4f}" + "".join(f"{d[i][j-i]:12.4f}" for i in range(j + 1)))

xn = x0 + (n - 1) * h
l = int(input("\nEnter the number of values to interpolate: "))
for _ in range(l):
    X = float(input("Enter x: "))
    v = (X - xn) / h
    Y, p = d[0][n-1], 1.0
    for i in range(1, n):
        p = p * (v + i - 1) / i
        Y = Y + p * d[i][n-1-i]
    print(f"v = {v:.4f}, P({X:g}) = {Y:.6f}")


OUTPUT

Enter the first value of x (x0): 1
Enter the step length h: 1
Enter the number of data points: 3
Enter the 3 values of y: 1 4 9

Backward difference table:
  1.0000      1.0000
  2.0000      4.0000      3.0000
  3.0000      9.0000      5.0000      2.0000

Enter the number of values to interpolate: 2
Enter x: 2.5
v = -0.5000, P(2.5) = 6.250000
Enter x: 2.8
v = -0.2000, P(2.8) = 7.840000


11.CUBIC SPLINE


import matplotlib.pyplot as plt

N = int(input("Enter the number of data points: "))
print("Enter t and y on each line:")
pts = [list(map(float, input().split())) for _ in range(N)]
t = [p[0] for p in pts]
y = [p[1] for p in pts]

n = N - 1
h = [t[i+1] - t[i] for i in range(n)]
b = [(y[i+1] - y[i]) / h[i] for i in range(n)]
u, v, z = [0.0] * n, [0.0] * n, [0.0] * (n + 1)
u[1], v[1] = 2 * (h[0] + h[1]), 6 * (b[1] - b[0])
for i in range(2, n):
    u[i] = 2 * (h[i-1] + h[i]) - h[i-1] ** 2 / u[i-1]
    v[i] = 6 * (b[i] - b[i-1]) - h[i-1] * v[i-1] / u[i-1]
for i in range(n - 1, 0, -1):
    z[i] = (v[i] - h[i] * z[i+1]) / u[i]

def S(i, d):
    B = -h[i] * z[i+1] / 6 - h[i] * z[i] / 3 + b[i]
    return y[i] + d * (B + d * (z[i] / 2 + d * (z[i+1] - z[i]) / (6 * h[i])))

print("\n       t           y           z")
for i in range(N):
    print(f"{t[i]:10.4f}  {y[i]:10.4f}  {z[i]:10.4f}")

l = int(input("\nEnter the number of values to interpolate: "))
for _ in range(l):
    X = float(input("Enter x: "))
    i = max(k for k in range(n) if t[k] <= X)
    print(f"S({X:g}) = {S(i, X - t[i]):.6f}")

for i in range(n):
    D = [h[i] * k / 50 for k in range(51)]
    plt.plot([t[i] + d for d in D], [S(i, d) for d in D], "b-")
plt.plot(t, y, "ko")
plt.title("Natural Cubic Spline")
plt.grid(True)
plt.show()


OUTPUT

Enter the number of data points: 3
Enter t and y on each line:
1 1
2 4
3 9

       t           y           z
    1.0000      1.0000      0.0000
    2.0000      4.0000      3.0000
    3.0000      9.0000      0.0000

Enter the number of values to interpolate: 2
Enter x: 1.5
S(1.5) = 2.312500
Enter x: 2.5
S(2.5) = 6.312500


12.TRAPEZOIDAL


from math import *

fx = input("Enter f(x): ")
a = float(input("Enter lower limit a: "))
b = float(input("Enter upper limit b: "))
n = int(input("Enter the number of subintervals n: "))
f = lambda x: eval(fx)
h = (b - a) / n

I = (h / 2) * (f(a) + f(b) + 2 * sum(f(a + i * h) for i in range(1, n)))
print(f"Integral by Trapezoidal rule = {I:.6f}")


OUTPUT

Enter f(x): x**2
Enter lower limit a: 0
Enter upper limit b: 2
Enter the number of subintervals n: 4
Integral by Trapezoidal rule = 2.750000


13.SIMPSON


from math import *

fx = input("Enter f(x): ")
a = float(input("Enter lower limit a: "))
b = float(input("Enter upper limit b: "))
n = int(input("Enter the number of subintervals n: "))
f = lambda x: eval(fx)
h = (b - a) / n

if n % 2 != 0:
    exit("n must be even")
odd = sum(f(a + i * h) for i in range(1, n, 2))
even = sum(f(a + i * h) for i in range(2, n, 2))
I = (h / 3) * (f(a) + f(b) + 4 * odd + 2 * even)
print(f"Integral by Simpson's 1/3 rule = {I:.6f}")


OUTPUT

Enter f(x): x**2
Enter lower limit a: 0
Enter upper limit b: 2
Enter the number of subintervals n: 4
Integral by Simpson's 1/3 rule = 2.666667


14.QUADRATURE


from math import *

fx = input("Enter f(x): ")
a = float(input("Enter lower limit a: "))
b = float(input("Enter upper limit b: "))
n = int(input("Enter the number of subintervals n: "))
f = lambda x: eval(fx)
h = (b - a) / n

d = h / (2 * sqrt(3))
I = 0.0
for i in range(n):
    m = a + i * h + h / 2
    I = I + (h / 2) * (f(m - d) + f(m + d))
print(f"Integral by Two-Point Gauss Quadrature = {I:.6f}")


OUTPUT

Enter f(x): x**2
Enter lower limit a: 0
Enter upper limit b: 2
Enter the number of subintervals n: 4
Integral by Two-Point Gauss Quadrature = 2.666667


15.ADAM BASHFORTH


from math import *

fs = input("Enter f(x, y): ")
f = lambda x, y: eval(fs)
print("Enter the 4 starting values (x y), one pair per line:")
pts = [list(map(float, input().split())) for _ in range(4)]
x = [p[0] for p in pts]
y = [p[1] for p in pts]
X_req = float(input("Enter the value of x at which y is required: "))
h = x[1] - x[0]
n = round((X_req - x[3]) / h)

for _ in range(n):
    F = [f(x[j], y[j]) for j in range(-4, 0)]
    xn = x[-1] + h
    yp = y[-1] + (h / 24) * (55 * F[3] - 59 * F[2] + 37 * F[1] - 9 * F[0])
    print(f"\nx = {xn:.4f}   Predicted y = {yp:.6f}")
    while True:
        yc = y[-1] + (h / 24) * (9 * f(xn, yp) + 19 * F[3] - 5 * F[2] + F[1])
        print(f"Corrected y = {yc:.6f}")
        if abs(yc - yp) < 1e-6:
            break
        yp = yc
    x.append(xn)
    y.append(yc)

print(f"\ny({X_req:g}) = {y[-1]:.6f}")


OUTPUT

Enter f(x, y): x**2 + 2*y
Enter the 4 starting values (x y), one pair per line:
0 1
0.1 1.2214
0.2 1.4932
0.3 1.8279
Enter the value of x at which y is required: 0.5

x = 0.4000   Predicted y = 2.245462
Corrected y = 2.246022
Corrected y = 2.246064
Corrected y = 2.246067
Corrected y = 2.246068

x = 0.5000   Predicted y = 2.765929
Corrected y = 2.765754
Corrected y = 2.765741
Corrected y = 2.765740

y(0.5) = 2.765740


16.MILNE


from math import *

fs = input("Enter f(x, y): ")
f = lambda x, y: eval(fs)
print("Enter the 4 starting values (x y), one pair per line:")
pts = [list(map(float, input().split())) for _ in range(4)]
x = [p[0] for p in pts]
y = [p[1] for p in pts]
X_req = float(input("Enter the value of x at which y is required: "))
h = x[1] - x[0]
n = round((X_req - x[3]) / h)

for _ in range(n):
    F = [f(x[j], y[j]) for j in range(-4, 0)]
    xn = x[-1] + h
    yp = y[-4] + (4 * h / 3) * (2 * F[1] - F[2] + 2 * F[3])
    print(f"\nx = {xn:.4f}   Predicted y = {yp:.6f}")
    while True:
        yc = y[-2] + (h / 3) * (F[2] + 4 * F[3] + f(xn, yp))
        print(f"Corrected y = {yc:.6f}")
        if abs(yc - yp) < 1e-6:
            break
        yp = yc
    x.append(xn)
    y.append(yc)

print(f"\ny({X_req:g}) = {y[-1]:.6f}")


OUTPUT

Enter f(x, y): x**2 + 2*y
Enter the 4 starting values (x y), one pair per line:
0 1
0.1 1.2214
0.2 1.4932
0.3 1.8279
Enter the value of x at which y is required: 0.5

x = 0.4000   Predicted y = 2.249440
Corrected y = 2.248816
Corrected y = 2.248774
Corrected y = 2.248772
Corrected y = 2.248771

x = 0.5000   Predicted y = 2.771011
Corrected y = 2.766833
Corrected y = 2.766555
Corrected y = 2.766536
Corrected y = 2.766535
Corrected y = 2.766535

y(0.5) = 2.766535


17.RUNGE KUTTA


from math import *

fs = input("Enter f(x, y): ")
f = lambda x, y: eval(fs)
x = float(input("Enter the initial value of x (x0): "))
y = float(input("Enter the initial value of y (y0): "))
h = float(input("Enter the step size (h): "))
X_req = float(input("Enter the value of x at which y is required: "))

for _ in range(round((X_req - x) / h)):
    k1 = h * f(x, y)
    k2 = h * f(x + h / 2, y + k1 / 2)
    k3 = h * f(x + h / 2, y + k2 / 2)
    k4 = h * f(x + h, y + k3)
    y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    x = x + h
    print(f"x = {x:.4f}   y = {y:.6f}")


OUTPUT

Enter f(x, y): x + y
Enter the initial value of x (x0): 0
Enter the initial value of y (y0): 1
Enter the step size (h): 0.1
Enter the value of x at which y is required: 0.5
x = 0.1000   y = 1.110342
x = 0.2000   y = 1.242805
x = 0.3000   y = 1.399717
x = 0.4000   y = 1.583648
x = 0.5000   y = 1.797441


18.FINITE DIFF FOR ELIPTICAL-method(1)


import numpy as np

n = int(input("Enter number of intervals n: "))
u = np.zeros((n + 1, n + 1))
u[0] = list(map(float, input(f"Top boundary ({n+1} values): ").split()))
u[n] = list(map(float, input(f"Bottom boundary ({n+1} values): ").split()))
u[1:n, 0] = list(map(float, input(f"Left boundary, top to bottom ({n-1} values): ").split()))
u[1:n, n] = list(map(float, input(f"Right boundary, top to bottom ({n-1} values): ").split()))

for it in range(1, 10001):
    old = u.copy()
    for i in range(1, n):
        for j in range(1, n):
            u[i, j] = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]) / 4
    if np.max(np.abs(u - old)) < 1e-6:
        break

print("\nSolution after", it, "iterations:")
print(np.round(u, 4))


OUTPUT

Enter number of intervals n: 4
Top boundary (5 values): 100 100 100 100 100
Bottom boundary (5 values): 0 0 0 0 0
Left boundary, top to bottom (3 values): 0 0 0
Right boundary, top to bottom (3 values): 0 0 0

Solution after 25 iterations:
[[100.     100.     100.     100.     100.    ]
 [  0.      42.8571  52.6786  42.8571   0.    ]
 [  0.      18.75    25.      18.75     0.    ]
 [  0.       7.1429   9.8214   7.1429   0.    ]
 [  0.       0.       0.       0.       0.    ]]


18.FINITE DIFF FOR ELIPTICAL-method(2)


import numpy as np

N = int(input("Enter number of intervals in x-direction (N): "))
M = int(input("Enter number of intervals in y-direction (M): "))
a, b = map(float, input("Enter x range a b: ").split())
c, d = map(float, input("Enter y range c d: ").split())
bottom = float(input("Enter bottom boundary value: "))
top = float(input("Enter top boundary value: "))
left = float(input("Enter left boundary value: "))
right = float(input("Enter right boundary value: "))
tol = float(input("Enter tolerance: "))
max_it = int(input("Enter maximum iterations: "))
print(f"\nh = {(b - a) / N}, k = {(d - c) / M}")

U = np.zeros((M + 1, N + 1))
U[0, :], U[M, :], U[:, 0], U[:, N] = bottom, top, left, right
U[0, 0], U[0, N] = (bottom + left) / 2, (bottom + right) / 2
U[M, 0], U[M, N] = (top + left) / 2, (top + right) / 2

for it in range(1, max_it + 1):
    old = U.copy()
    for j in range(1, M):
        for i in range(1, N):
            U[j, i] = (U[j, i+1] + U[j, i-1] + U[j+1, i] + U[j-1, i]) / 4
    E = np.max(np.abs(U - old))
    print(f"Iteration {it}: maximum change E = {E:.6f}")
    if E < tol:
        break

print("\nFinal solution (top row is y = d):")
for row in U[::-1]:
    print(" ".join(f"{v:10.4f}" for v in row))


OUTPUT

Enter number of intervals in x-direction (N): 4
Enter number of intervals in y-direction (M): 4
Enter x range a b: 0 4
Enter y range c d: 0 4
Enter bottom boundary value: 0
Enter top boundary value: 100
Enter left boundary value: 0
Enter right boundary value: 0
Enter tolerance: 0.01
Enter maximum iterations: 100

h = 1.0, k = 1.0
Iteration 1: maximum change E = 32.812500
Iteration 2: maximum change E = 12.890625
Iteration 3: maximum change E = 7.812500
Iteration 4: maximum change E = 3.906250
Iteration 5: maximum change E = 1.953125
Iteration 6: maximum change E = 0.976562
Iteration 7: maximum change E = 0.488281
Iteration 8: maximum change E = 0.244141
Iteration 9: maximum change E = 0.122070
Iteration 10: maximum change E = 0.061035
Iteration 11: maximum change E = 0.030518
Iteration 12: maximum change E = 0.015259
Iteration 13: maximum change E = 0.007629

Final solution (top row is y = d):
   50.0000   100.0000   100.0000   100.0000    50.0000
    0.0000    42.8533    52.6748    42.8552     0.0000
    0.0000    18.7424    24.9924    18.7462     0.0000
    0.0000     7.1352     9.8138     7.1390     0.0000
    0.0000     0.0000     0.0000     0.0000     0.0000


19.FINITE DIFFERENCE PARABOLIC(METHOD-1)


from math import *

L = float(input("Enter length of the rod L: "))
T = float(input("Enter total time T: "))
alpha = float(input("Enter thermal diffusivity alpha (c^2): "))
nx = int(input("Enter number of space intervals nx: "))
nt = int(input("Enter number of time intervals nt: "))
h, k = L / nx, T / nt
r = alpha * k / h ** 2
print(f"h = {h:.4f}, k = {k:.4f}, r = {r:.4f}")
if r > 0.5:
    print("Warning: r > 0.5, the explicit scheme is unstable")
fs = input("Enter initial temperature f(x): ")
f = lambda x: eval(fs)
left = float(input("Enter left boundary temperature: "))
right = float(input("Enter right boundary temperature: "))

u = [[0.0] * (nx + 1) for _ in range(nt + 1)]
u[0] = [f(i * h) for i in range(nx + 1)]
for j in range(nt + 1):
    u[j][0], u[j][nx] = left, right
for j in range(nt):
    for i in range(1, nx):
        u[j+1][i] = r * u[j][i-1] + (1 - 2 * r) * u[j][i] + r * u[j][i+1]

print("\n   t \\ x" + "".join(f"{i * h:10.4f}" for i in range(nx + 1)))
for j in range(nt + 1):
    print(f"{j * k:8.4f}" + "".join(f"{v:10.4f}" for v in u[j]))


OUTPUT

Enter length of the rod L: 2
Enter total time T: 1
Enter thermal diffusivity alpha (c^2): 1
Enter number of space intervals nx: 2
Enter number of time intervals nt: 4
h = 1.0000, k = 0.2500, r = 0.2500
Enter initial temperature f(x): x*(2-x)
Enter left boundary temperature: 0
Enter right boundary temperature: 0

   t \ x    0.0000    1.0000    2.0000
  0.0000    0.0000    1.0000    0.0000
  0.2500    0.0000    0.5000    0.0000
  0.5000    0.0000    0.2500    0.0000
  0.7500    0.0000    0.1250    0.0000
  1.0000    0.0000    0.0625    0.0000


19.FINITE DIFFERENCE PARABOLIC(METHOD-2)-FTCS METHOD


import numpy as np

L = float(input("Enter length of the rod L: "))
T = float(input("Enter total time T: "))
alpha = float(input("Enter thermal diffusivity alpha (c^2): "))
nx = int(input("Enter number of space intervals nx: "))
nt = int(input("Enter number of time intervals nt: "))
h, k = L / nx, T / nt
r = alpha * k / h ** 2
print(f"h = {h:.4f}, k = {k:.4f}, r = {r:.4f}")
if r > 0.5:
    exit("Stability condition r <= 0.5 violated")
u = np.zeros((nt + 1, nx + 1))
u[0] = list(map(float, input(f"Enter the {nx + 1} initial temperatures: ").split()))
u[:, 0] = float(input("Enter left boundary temperature: "))
u[:, nx] = float(input("Enter right boundary temperature: "))

for j in range(nt):
    for i in range(1, nx):
        u[j+1][i] = u[j][i] + r * (u[j][i+1] - 2 * u[j][i] + u[j][i-1])

print("\n   t \\ x" + "".join(f"{i * h:10.4f}" for i in range(nx + 1)))
for j in range(nt + 1):
    print(f"{j * k:8.4f}" + "".join(f"{v:10.4f}" for v in u[j]))


OUTPUT

Enter length of the rod L: 2
Enter total time T: 1
Enter thermal diffusivity alpha (c^2): 1
Enter number of space intervals nx: 2
Enter number of time intervals nt: 4
h = 1.0000, k = 0.2500, r = 0.2500
Enter the 3 initial temperatures: 0 1 0
Enter left boundary temperature: 0
Enter right boundary temperature: 0

   t \ x    0.0000    1.0000    2.0000
  0.0000    0.0000    1.0000    0.0000
  0.2500    0.0000    0.5000    0.0000
  0.5000    0.0000    0.2500    0.0000
  0.7500    0.0000    0.1250    0.0000
  1.0000    0.0000    0.0625    0.0000


19.FINITE DIFFERENCE PARABOLIC(METHOD-3)-CRANK NICOLSON METHOD


import numpy as np

L = float(input("Enter length of the rod L: "))
T = float(input("Enter total time T: "))
alpha = float(input("Enter thermal diffusivity alpha (c^2): "))
nx = int(input("Enter number of space intervals nx: "))
nt = int(input("Enter number of time intervals nt: "))
h, k = L / nx, T / nt
r = alpha * k / h ** 2
print(f"h = {h:.4f}, k = {k:.4f}, r = {r:.4f}")
u = np.zeros((nt + 1, nx + 1))
u[0] = list(map(float, input(f"Enter the {nx + 1} initial temperatures: ").split()))
u[:, 0] = float(input("Enter left boundary temperature: "))
u[:, nx] = float(input("Enter right boundary temperature: "))

m = nx - 1
A = np.diag([2 + 2 * r] * m) + np.diag([-r] * (m - 1), 1) + np.diag([-r] * (m - 1), -1)
for j in range(nt):
    rhs = [r * u[j][i-1] + (2 - 2 * r) * u[j][i] + r * u[j][i+1] for i in range(1, nx)]
    rhs[0] += r * u[j+1][0]
    rhs[-1] += r * u[j+1][nx]
    u[j+1, 1:nx] = np.linalg.solve(A, rhs)

print("\n   t \\ x" + "".join(f"{i * h:10.4f}" for i in range(nx + 1)))
for j in range(nt + 1):
    print(f"{j * k:8.4f}" + "".join(f"{v:10.4f}" for v in u[j]))


OUTPUT

Enter length of the rod L: 2
Enter total time T: 1
Enter thermal diffusivity alpha (c^2): 1
Enter number of space intervals nx: 2
Enter number of time intervals nt: 4
h = 1.0000, k = 0.2500, r = 0.2500
Enter the 3 initial temperatures: 0 1 0
Enter left boundary temperature: 0
Enter right boundary temperature: 0

   t \ x    0.0000    1.0000    2.0000
  0.0000    0.0000    1.0000    0.0000
  0.2500    0.0000    0.6000    0.0000
  0.5000    0.0000    0.3600    0.0000
  0.7500    0.0000    0.2160    0.0000
  1.0000    0.0000    0.1296    0.0000


20.FINITE DIFF WITH HYPERBOLIC(METHOD1)


import numpy as np

L = float(input("Enter length of the string L: "))
T = float(input("Enter total time T: "))
c = float(input("Enter wave speed c: "))
nx = int(input("Enter number of space intervals nx: "))
nt = int(input("Enter number of time intervals nt: "))
h, k = L / nx, T / nt
r = c * k / h
print(f"h = {h:.4f}, k = {k:.4f}, r = {r:.4f}")
if r > 1:
    exit("Stability condition r <= 1 violated")
u = np.zeros((nt + 1, nx + 1))
u[0] = list(map(float, input(f"Enter the {nx + 1} initial displacements: ").split()))

for i in range(1, nx):
    u[1][i] = (1 - r**2) * u[0][i] + (r**2 / 2) * (u[0][i+1] + u[0][i-1])
for j in range(1, nt):
    for i in range(1, nx):
        u[j+1][i] = 2 * (1 - r**2) * u[j][i] + r**2 * (u[j][i+1] + u[j][i-1]) - u[j-1][i]

print("\n   t \\ x" + "".join(f"{i * h:10.4f}" for i in range(nx + 1)))
for j in range(nt + 1):
    print(f"{j * k:8.4f}" + "".join(f"{v:10.4f}" for v in u[j]))


OUTPUT

Enter length of the string L: 2
Enter total time T: 1
Enter wave speed c: 1
Enter number of space intervals nx: 2
Enter number of time intervals nt: 4
h = 1.0000, k = 0.2500, r = 0.2500
Enter the 3 initial displacements: 0 1 0

   t \ x    0.0000    1.0000    2.0000
  0.0000    0.0000    1.0000    0.0000
  0.2500    0.0000    0.9375    0.0000
  0.5000    0.0000    0.7578    0.0000
  0.7500    0.0000    0.4834    0.0000
  1.0000    0.0000    0.1486    0.0000


20.FINITE DIFF WITH HYPERBOLIC(METHOD2)


from math import *

L = float(input("Enter length of the string L: "))
T = float(input("Enter total time T: "))
c = float(input("Enter wave speed c: "))
nx = int(input("Enter number of space intervals nx: "))
nt = int(input("Enter number of time intervals nt: "))
h, k = L / nx, T / nt
r = c * k / h
print(f"h = {h:.4f}, k = {k:.4f}, r = {r:.4f}")
if r > 1:
    exit("Stability condition r <= 1 violated")
fs = input("Enter initial displacement f(x): ")
gs = input("Enter initial velocity g(x): ")
f = lambda x: eval(fs)
g = lambda x: eval(gs)

u = [[0.0] * (nx + 1) for _ in range(nt + 1)]
u[0] = [f(i * h) for i in range(nx + 1)]
for i in range(1, nx):
    u[1][i] = u[0][i] + k * g(i * h) + (r**2 / 2) * (u[0][i+1] - 2 * u[0][i] + u[0][i-1])
for j in range(1, nt):
    for i in range(1, nx):
        u[j+1][i] = 2 * (1 - r**2) * u[j][i] + r**2 * (u[j][i+1] + u[j][i-1]) - u[j-1][i]

print("\n   t \\ x" + "".join(f"{i * h:10.4f}" for i in range(nx + 1)))
for j in range(nt + 1):
    print(f"{j * k:8.4f}" + "".join(f"{v:10.4f}" for v in u[j]))


OUTPUT

Enter length of the string L: 2
Enter total time T: 1
Enter wave speed c: 1
Enter number of space intervals nx: 2
Enter number of time intervals nt: 4
h = 1.0000, k = 0.2500, r = 0.2500
Enter initial displacement f(x): x*(2-x)
Enter initial velocity g(x): 0

   t \ x    0.0000    1.0000    2.0000
  0.0000    0.0000    1.0000    0.0000
  0.2500    0.0000    0.9375    0.0000
  0.5000    0.0000    0.7578    0.0000
  0.7500    0.0000    0.4834    0.0000
  1.0000    0.0000    0.1486    0.0000
