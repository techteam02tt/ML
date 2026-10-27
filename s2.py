"""
S2.PY - NUMERICAL METHODS REFERENCE
===================================

Programs included:
01. Gauss Elimination
02. Gauss Elimination with Row Pivoting
03. Gauss-Jacobi Method
04. Gauss-Seidel Method
05. Power Method
06. Gauss-Jordan Matrix Inverse
07. Lagrange Interpolation
08. Newton Divided Difference
09. Newton Forward Difference
10. Newton Backward Difference
11. Natural Cubic Spline
12. Composite Trapezoidal Rule
13. Composite Simpson's 1/3 Rule
14. Composite Two-Point Gauss Quadrature
15. Adams-Bashforth-Moulton Predictor-Corrector
16. Milne Predictor-Corrector
17. Runge-Kutta Fourth-Order Method
18. Finite Difference Method for Elliptic PDE

Install optional libraries:
    pip install sympy matplotlib numpy

Run:
    python3 s2.py

Select a method from the menu.
"""

import math


# ---------------------------------------------------------------------------
# 01. GAUSS ELIMINATION
# ---------------------------------------------------------------------------

def gauss_elimination():
    n = int(input("Enter the order of the matrix: "))
    print("Enter the coefficient matrix A:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)]
    b = list(map(float, input("Enter the RHS vector b: ").split()))

    if len(b) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix and RHS dimensions do not match.")

    for k in range(n - 1):
        if abs(A[k][k]) < 1e-12:
            raise ValueError("Zero pivot. Use the row-pivoting method.")
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
            A[i][k] = 0.0

    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        if abs(A[k][k]) < 1e-12:
            raise ValueError("Singular matrix; unique solution does not exist.")
        s = sum(A[k][j] * x[j] for j in range(k + 1, n))
        x[k] = (b[k] - s) / A[k][k]

    print("\nSolution vector:")
    for i, value in enumerate(x, 1):
        print(f"x{i} = {value:.6f}")


# ---------------------------------------------------------------------------
# 02. GAUSS ELIMINATION WITH ROW PIVOTING
# ---------------------------------------------------------------------------

def gauss_elimination_pivoting():
    n = int(input("Enter the order of the matrix: "))
    print("Enter the coefficient matrix A:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)]
    b = list(map(float, input("Enter the RHS vector b: ").split()))
    tol = 1e-12

    if len(b) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix and RHS dimensions do not match.")

    for k in range(n - 1):
        p = max(range(k, n), key=lambda row: abs(A[row][k]))
        if abs(A[p][k]) <= tol:
            raise ValueError("Singular matrix; unique solution does not exist.")
        if p != k:
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
            print(f"Pivot is zero/small; rows {k + 1} and {p + 1} swapped.")

        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
            A[i][k] = 0.0

    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        if abs(A[k][k]) <= tol:
            raise ValueError("Singular matrix; unique solution does not exist.")
        s = sum(A[k][j] * x[j] for j in range(k + 1, n))
        x[k] = (b[k] - s) / A[k][k]

    print("\nSolution vector:")
    for i, value in enumerate(x, 1):
        print(f"x{i} = {value:.6f}")


# ---------------------------------------------------------------------------
# 03. GAUSS-JACOBI METHOD
# ---------------------------------------------------------------------------

def gauss_jacobi():
    n = int(input("Enter the order of the matrix: "))
    print("Enter the coefficient matrix A:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)]
    b = list(map(float, input("Enter the RHS vector b: ").split()))
    x_old = list(map(float, input("Enter the initial guess x0: ").split()))
    tol = float(input("Enter tolerance TOL: "))
    kmax = int(input("Enter maximum iterations KMAX: "))

    if any(len(row) != n for row in A) or len(b) != n or len(x_old) != n:
        raise ValueError("Matrix, RHS, and initial guess dimensions must match.")
    if any(abs(A[i][i]) < 1e-12 for i in range(n)):
        raise ValueError("Diagonal elements must be non-zero.")

    for k in range(1, kmax + 1):
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        error = max(abs(x_new[i] - x_old[i]) for i in range(n))
        print(f"Iteration {k}: {[round(v, 6) for v in x_new]} Error = {error:.6f}")
        if error <= tol:
            print(f"\nConverged after {k} iterations.")
            x_old = x_new
            break
        x_old = x_new
    else:
        print("\nMaximum iterations reached.")

    print("Solution:")
    for i, value in enumerate(x_old, 1):
        print(f"x{i} = {value:.6f}")


# ---------------------------------------------------------------------------
# 04. GAUSS-SEIDEL METHOD
# ---------------------------------------------------------------------------

def gauss_seidel():
    n = int(input("Enter the order of the matrix: "))
    print("Enter the coefficient matrix A:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)]
    b = list(map(float, input("Enter the RHS vector b: ").split()))
    x_old = list(map(float, input("Enter the initial guess x0: ").split()))
    tol = float(input("Enter tolerance TOL: "))
    kmax = int(input("Enter maximum iterations KMAX: "))

    if any(len(row) != n for row in A) or len(b) != n or len(x_old) != n:
        raise ValueError("Matrix, RHS, and initial guess dimensions must match.")
    if any(abs(A[i][i]) < 1e-12 for i in range(n)):
        raise ValueError("Diagonal elements must be non-zero.")

    for k in range(1, kmax + 1):
        x_new = x_old.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        error = max(abs(x_new[i] - x_old[i]) for i in range(n))
        print(f"Iteration {k}: {[round(v, 6) for v in x_new]} Error = {error:.6f}")
        x_old = x_new
        if error <= tol:
            print(f"\nConverged after {k} iterations.")
            break
    else:
        print("\nMaximum iterations reached.")

    print("Solution:")
    for i, value in enumerate(x_old, 1):
        print(f"x{i} = {value:.6f}")


# ---------------------------------------------------------------------------
# 05. POWER METHOD
# ---------------------------------------------------------------------------

def power_method():
    n = int(input("Enter order of matrix: "))
    print("Enter matrix A:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)]
    v = list(map(float, input("Enter initial vector: ").split()))
    tol = float(input("Enter tolerance: "))
    kmax = int(input("Enter maximum iterations: "))

    if any(len(row) != n for row in A) or len(v) != n:
        raise ValueError("Matrix and vector dimensions must match.")
    if max(map(abs, v)) < 1e-14:
        raise ValueError("Initial vector must be non-zero.")

    lam_old = None
    for k in range(1, kmax + 1):
        w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        c = max(w, key=abs)
        if abs(c) < 1e-14:
            raise ValueError("Power method reached a zero vector.")
        v_new = [wi / c for wi in w]
        lam = c
        vector_error = max(abs(v_new[i] - v[i]) for i in range(n))
        eigen_error = abs(lam - lam_old) if lam_old is not None else float("inf")
        error = max(vector_error, eigen_error)
        v = v_new
        lam_old = lam
        print(f"Iteration {k}: Eigen Value = {lam:.6f}, "
              f"Eigen Vector = {[round(value, 6) for value in v]}")
        if error <= tol:
            break

    print(f"\nDominant Eigen Value estimate = {lam_old:.6f}")
    print("Dominant Eigen Vector =", [round(value, 6) for value in v])


# ---------------------------------------------------------------------------
# 06. GAUSS-JORDAN MATRIX INVERSE
# ---------------------------------------------------------------------------

def jordan_inverse():
    n = int(input("Enter the order of the matrix: "))
    print("Enter matrix A:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)]
    if any(len(row) != n for row in A):
        raise ValueError("Enter a square matrix.")

    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]
    tol = 1e-12

    for k in range(n):
        p = max(range(k, n), key=lambda row: abs(aug[row][k]))
        if abs(aug[p][k]) < tol:
            raise ValueError("Matrix is singular. Inverse does not exist.")
        aug[k], aug[p] = aug[p], aug[k]
        pivot = aug[k][k]
        aug[k] = [value / pivot for value in aug[k]]

        for i in range(n):
            if i != k:
                factor = aug[i][k]
                aug[i] = [aug[i][j] - factor * aug[k][j]
                          for j in range(2 * n)]

    print("\nInverse matrix:")
    for row in aug:
        print(" ".join(f"{value:9.4f}" for value in row[n:]))


# ---------------------------------------------------------------------------
# 07. LAGRANGE INTERPOLATION
# ---------------------------------------------------------------------------

def lagrange_interpolation():
    import sympy as sp

    n = int(input("Enter the number of data points: "))
    print("Enter x and f(x) on each line:")
    pts = [list(map(float, input().split())) for _ in range(n)]
    if any(len(point) != 2 for point in pts):
        raise ValueError("Each point must contain x and y.")
    if len({point[0] for point in pts}) != n:
        raise ValueError("All x values must be distinct.")

    xt = float(input("Enter the value of x to interpolate: "))
    X = sp.Symbol("x")
    P = 0
    value = 0.0
    ops = 0
    terms = []

    for i in range(n):
        L = 1
        p = 1.0
        numerator_parts = []
        denominator_parts = []
        for j in range(n):
            if j != i:
                L *= (X - pts[j][0]) / (pts[i][0] - pts[j][0])
                p *= (xt - pts[j][0]) / (pts[i][0] - pts[j][0])
                numerator_parts.append(f"(x-{pts[j][0]:g})")
                denominator_parts.append(f"({pts[i][0]:g}-{pts[j][0]:g})")
                ops += 4
        P += L * pts[i][1]
        value += p * pts[i][1]
        ops += 1
        terms.append(f"{pts[i][1]:g}*{''.join(numerator_parts)}/"
                     f"{''.join(denominator_parts)}")

    poly = sp.nsimplify(sp.expand(P))
    coefficients = sp.Poly(poly, X).all_coeffs()
    horner_value = float(coefficients[0])
    for coefficient in coefficients[1:]:
        horner_value = horner_value * xt + float(coefficient)

    print("\n(a) Lagrange form: P(x) =", " + ".join(terms))
    print("(b) Standard form: P(x) =", poly)
    print(f"(c) P({xt:g}) using Lagrange form = {value:.6f}, operations = {ops}")
    print("(d) Operations to convert to standard form =", sp.count_ops(poly))
    print(f"(e) P({xt:g}) using Horner's method = {horner_value:.6f}, "
          f"operations = {2 * (len(coefficients) - 1)}")


# ---------------------------------------------------------------------------
# 08. NEWTON DIVIDED DIFFERENCE
# ---------------------------------------------------------------------------

def newton_divided_difference():
    n = int(input("Enter the number of data points: "))
    print("Enter x and y on each line:")
    pts = [list(map(float, input().split())) for _ in range(n)]
    x = [point[0] for point in pts]
    y = [point[1] for point in pts]
    if len(set(x)) != n:
        raise ValueError("All x values must be distinct.")

    d = [[0.0] * n for _ in range(n)]
    d[0] = y[:]
    for i in range(1, n):
        for j in range(n - i):
            d[i][j] = (d[i - 1][j + 1] - d[i - 1][j]) / (x[j + i] - x[j])

    print("\nDivided difference table:")
    for j in range(n):
        print(f"{x[j]:8.4f}" +
              "".join(f"{d[i][j]:12.4f}" for i in range(n - j)))

    print("\nNewton interpolation coefficients:", [d[i][0] for i in range(n)])
    count = int(input("\nEnter the number of values to interpolate: "))
    for _ in range(count):
        X = float(input("Enter x: "))
        Y, product = d[0][0], 1.0
        for i in range(n - 1):
            product *= X - x[i]
            Y += product * d[i + 1][0]
        print(f"P({X:g}) = {Y:.6f}")


# ---------------------------------------------------------------------------
# 09. NEWTON FORWARD DIFFERENCE
# ---------------------------------------------------------------------------

def forward_difference():
    x0 = float(input("Enter the first value of x (x0): "))
    h = float(input("Enter the step length h: "))
    n = int(input("Enter the number of data points: "))
    if n < 2 or abs(h) < 1e-12:
        raise ValueError("At least two points and a non-zero step are required.")
    y = list(map(float, input(f"Enter the {n} values of y: ").split()))
    if len(y) != n:
        raise ValueError("Incorrect number of y values.")

    d = [[0.0] * n for _ in range(n)]
    d[0] = y[:]
    for i in range(1, n):
        for j in range(n - i):
            d[i][j] = d[i - 1][j + 1] - d[i - 1][j]

    print("\nForward difference table:")
    for j in range(n):
        print(f"{x0 + j * h:8.4f}" +
              "".join(f"{d[i][j]:12.4f}" for i in range(n - j)))

    count = int(input("\nEnter the number of values to interpolate: "))
    for _ in range(count):
        X = float(input("Enter x: "))
        u = (X - x0) / h
        Y, product = d[0][0], 1.0
        for i in range(1, n):
            product *= (u - i + 1) / i
            Y += product * d[i][0]
        print(f"u = {u:.4f}, P({X:g}) = {Y:.6f}")


# ---------------------------------------------------------------------------
# 10. NEWTON BACKWARD DIFFERENCE
# ---------------------------------------------------------------------------

def backward_difference():
    x0 = float(input("Enter the first value of x (x0): "))
    h = float(input("Enter the step length h: "))
    n = int(input("Enter the number of data points: "))
    if n < 2 or abs(h) < 1e-12:
        raise ValueError("At least two points and a non-zero step are required.")
    y = list(map(float, input(f"Enter the {n} values of y: ").split()))
    if len(y) != n:
        raise ValueError("Incorrect number of y values.")

    d = [[0.0] * n for _ in range(n)]
    d[0] = y[:]
    for i in range(1, n):
        for j in range(n - i):
            d[i][j] = d[i - 1][j + 1] - d[i - 1][j]

    print("\nBackward difference table:")
    for j in range(n):
        print(f"{x0 + j * h:8.4f}" +
              "".join(f"{d[i][j - i]:12.4f}" for i in range(j + 1)))

    xn = x0 + (n - 1) * h
    count = int(input("\nEnter the number of values to interpolate: "))
    for _ in range(count):
        X = float(input("Enter x: "))
        v = (X - xn) / h
        Y, product = d[0][n - 1], 1.0
        for i in range(1, n):
            product *= (v + i - 1) / i
            Y += product * d[i][n - 1 - i]
        print(f"v = {v:.4f}, P({X:g}) = {Y:.6f}")


# ---------------------------------------------------------------------------
# 11. NATURAL CUBIC SPLINE
# ---------------------------------------------------------------------------

def cubic_spline():
    import matplotlib.pyplot as plt

    N = int(input("Enter the number of data points (at least 3): "))
    if N < 3:
        raise ValueError("At least 3 data points are required.")
    print("Enter t and y on each line:")
    pts = [list(map(float, input().split())) for _ in range(N)]
    t = [point[0] for point in pts]
    y = [point[1] for point in pts]
    if any(t[i] <= t[i - 1] for i in range(1, N)):
        raise ValueError("The t values must be strictly increasing.")

    n = N - 1
    h = [t[i + 1] - t[i] for i in range(n)]
    slope = [(y[i + 1] - y[i]) / h[i] for i in range(n)]
    z = [0.0] * (n + 1)

    # Solve the tridiagonal system for the interior second derivatives.
    if n > 1:
        lower = [0.0] * (n - 1)
        diagonal = [0.0] * (n - 1)
        upper = [0.0] * (n - 1)
        rhs = [0.0] * (n - 1)
        for row, i in enumerate(range(1, n)):
            lower[row] = h[i - 1] if row > 0 else 0.0
            diagonal[row] = 2 * (h[i - 1] + h[i])
            upper[row] = h[i] if row < n - 2 else 0.0
            rhs[row] = 6 * (slope[i] - slope[i - 1])

        # Thomas algorithm
        for i in range(1, len(diagonal)):
            factor = lower[i] / diagonal[i - 1]
            diagonal[i] -= factor * upper[i - 1]
            rhs[i] -= factor * rhs[i - 1]
        interior = [0.0] * len(diagonal)
        interior[-1] = rhs[-1] / diagonal[-1]
        for i in range(len(interior) - 2, -1, -1):
            interior[i] = (rhs[i] - upper[i] * interior[i + 1]) / diagonal[i]
        for i, value in enumerate(interior, 1):
            z[i] = value

    def S(i, delta):
        return (y[i]
                + delta * (slope[i] - h[i] * (2 * z[i] + z[i + 1]) / 6
                           + delta * (z[i] / 2
                                      + delta * (z[i + 1] - z[i]) / (6 * h[i]))))

    print("\n t          y          second derivative")
    for i in range(N):
        print(f"{t[i]:10.4f} {y[i]:10.4f} {z[i]:10.4f}")

    count = int(input("\nEnter the number of values to interpolate: "))
    for _ in range(count):
        X = float(input("Enter x: "))
        if not t[0] <= X <= t[-1]:
            print("Value is outside the spline interval.")
            continue
        i = next((k for k in range(n) if t[k] <= X <= t[k + 1]), n - 1)
        i = min(i, n - 1)
        print(f"S({X:g}) = {S(i, X - t[i]):.6f}")

    for i in range(n):
        sample_x = [t[i] + h[i] * k / 50 for k in range(51)]
        sample_y = [S(i, xv - t[i]) for xv in sample_x]
        plt.plot(sample_x, sample_y, "b-")
    plt.plot(t, y, "ko", label="Data points")
    plt.title("Natural Cubic Spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# 12. COMPOSITE TRAPEZOIDAL RULE
# ---------------------------------------------------------------------------

def trapezoidal_rule():
    from math import sin, cos, tan, exp, sqrt, log, pi, e

    fx = input("Enter f(x), e.g. x**2: ")
    a = float(input("Enter lower limit a: "))
    b = float(input("Enter upper limit b: "))
    n = int(input("Enter the number of subintervals n: "))
    if n < 1:
        raise ValueError("Number of subintervals must be positive.")

    def f(x):
        return eval(fx, {"__builtins__": {}},
                    {"x": x, "sin": sin, "cos": cos, "tan": tan,
                     "exp": exp, "sqrt": sqrt, "log": log, "pi": pi, "e": e})

    h = (b - a) / n
    I = (h / 2) * (f(a) + f(b)
                   + 2 * sum(f(a + i * h) for i in range(1, n)))
    print(f"Integral by Trapezoidal rule = {I:.6f}")


# ---------------------------------------------------------------------------
# 13. COMPOSITE SIMPSON'S 1/3 RULE
# ---------------------------------------------------------------------------

def simpsons_rule():
    from math import sin, cos, tan, exp, sqrt, log, pi, e

    fx = input("Enter f(x), e.g. x**2: ")
    a = float(input("Enter lower limit a: "))
    b = float(input("Enter upper limit b: "))
    n = int(input("Enter the number of subintervals n (must be even): "))
    if n < 2 or n % 2 != 0:
        raise ValueError("n must be a positive even integer.")

    def f(x):
        return eval(fx, {"__builtins__": {}},
                    {"x": x, "sin": sin, "cos": cos, "tan": tan,
                     "exp": exp, "sqrt": sqrt, "log": log, "pi": pi, "e": e})

    h = (b - a) / n
    odd = sum(f(a + i * h) for i in range(1, n, 2))
    even = sum(f(a + i * h) for i in range(2, n, 2))
    I = (h / 3) * (f(a) + f(b) + 4 * odd + 2 * even)
    print(f"Integral by Simpson's 1/3 rule = {I:.6f}")


# ---------------------------------------------------------------------------
# 14. COMPOSITE TWO-POINT GAUSS QUADRATURE
# ---------------------------------------------------------------------------

def gauss_quadrature():
    from math import sin, cos, tan, exp, sqrt, log, pi, e

    fx = input("Enter f(x), e.g. x**2: ")
    a = float(input("Enter lower limit a: "))
    b = float(input("Enter upper limit b: "))
    n = int(input("Enter the number of subintervals n: "))
    if n < 1:
        raise ValueError("Number of subintervals must be positive.")

    def f(x):
        return eval(fx, {"__builtins__": {}},
                    {"x": x, "sin": sin, "cos": cos, "tan": tan,
                     "exp": exp, "sqrt": sqrt, "log": log, "pi": pi, "e": e})

    h = (b - a) / n
    d = h / (2 * sqrt(3))
    I = 0.0
    for i in range(n):
        midpoint = a + i * h + h / 2
        I += (h / 2) * (f(midpoint - d) + f(midpoint + d))
    print(f"Integral by Two-Point Gauss Quadrature = {I:.6f}")


# ---------------------------------------------------------------------------
# HELPER FOR ODE METHODS
# ---------------------------------------------------------------------------

def evaluate_f(expression, x, y):
    """Evaluate a user-entered ODE expression such as x + y."""
    allowed = {
        "x": x, "y": y, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "exp": math.exp, "sqrt": math.sqrt,
        "log": math.log, "pi": math.pi, "e": math.e, "abs": abs
    }
    return float(eval(expression, {"__builtins__": {}}, allowed))


def read_ode_starting_values():
    print("Enter the 4 starting values (x y), one pair per line:")
    points = [list(map(float, input().split())) for _ in range(4)]
    if any(len(point) != 2 for point in points):
        raise ValueError("Each starting point must contain x and y.")
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    h = x[1] - x[0]
    if abs(h) < 1e-12 or any(abs((x[i] - x[i - 1]) - h) > 1e-8
                             for i in range(1, 4)):
        raise ValueError("Starting x values must be equally spaced.")
    return x, y, h


# ---------------------------------------------------------------------------
# 15. ADAMS-BASHFORTH-MOULTON PREDICTOR-CORRECTOR
# ---------------------------------------------------------------------------

def adams_bashforth_moulton():
    fs = input("Enter f(x, y), e.g. x**2 + 2*y: ")
    x, y, h = read_ode_starting_values()
    target = float(input("Enter the value of x at which y is required: "))
    steps_float = (target - x[-1]) / h
    steps = round(steps_float)
    if steps < 0 or abs(steps_float - steps) > 1e-8:
        raise ValueError("Target must be ahead of the last x by whole steps.")

    for _ in range(steps):
        F = [evaluate_f(fs, x[j], y[j]) for j in range(len(x) - 4, len(x))]
        xn = x[-1] + h
        yp = y[-1] + (h / 24) * (
            55 * F[3] - 59 * F[2] + 37 * F[1] - 9 * F[0]
        )
        print(f"\nx = {xn:.4f}, predicted y = {yp:.6f}")
        for _iteration in range(100):
            yc = y[-1] + (h / 24) * (
                9 * evaluate_f(fs, xn, yp) + 19 * F[3] - 5 * F[2] + F[1]
            )
            print(f"Corrected y = {yc:.6f}")
            if abs(yc - yp) < 1e-6:
                yp = yc
                break
            yp = yc
        x.append(xn)
        y.append(yp)

    print(f"\ny({target:g}) = {y[-1]:.6f}")


# ---------------------------------------------------------------------------
# 16. MILNE PREDICTOR-CORRECTOR
# ---------------------------------------------------------------------------

def milne_predictor_corrector():
    fs = input("Enter f(x, y), e.g. x**2 + 2*y: ")
    x, y, h = read_ode_starting_values()
    target = float(input("Enter the value of x at which y is required: "))
    steps_float = (target - x[-1]) / h
    steps = round(steps_float)
    if steps < 0 or abs(steps_float - steps) > 1e-8:
        raise ValueError("Target must be ahead of the last x by whole steps.")

    for _ in range(steps):
        F = [evaluate_f(fs, x[j], y[j]) for j in range(len(x) - 4, len(x))]
        xn = x[-1] + h
        yp = y[-4] + (4 * h / 3) * (2 * F[1] - F[2] + 2 * F[3])
        print(f"\nx = {xn:.4f}, predicted y = {yp:.6f}")
        for _iteration in range(100):
            yc = y[-2] + (h / 3) * (
                F[2] + 4 * F[3] + evaluate_f(fs, xn, yp)
            )
            print(f"Corrected y = {yc:.6f}")
            if abs(yc - yp) < 1e-6:
                yp = yc
                break
            yp = yc
        x.append(xn)
        y.append(yp)

    print(f"\ny({target:g}) = {y[-1]:.6f}")


# ---------------------------------------------------------------------------
# 17. RUNGE-KUTTA FOURTH-ORDER METHOD
# ---------------------------------------------------------------------------

def runge_kutta():
    fs = input("Enter f(x, y), e.g. x + y: ")
    x = float(input("Enter the initial value of x (x0): "))
    y = float(input("Enter the initial value of y (y0): "))
    h = float(input("Enter the step size (h): "))
    target = float(input("Enter the value of x at which y is required: "))
    if abs(h) < 1e-12:
        raise ValueError("Step size cannot be zero.")

    steps_float = (target - x) / h
    steps = round(steps_float)
    if steps < 0 or abs(steps_float - steps) > 1e-8:
        raise ValueError("Target must be reachable by whole steps.")

    for _ in range(steps):
        k1 = h * evaluate_f(fs, x, y)
        k2 = h * evaluate_f(fs, x + h / 2, y + k1 / 2)
        k3 = h * evaluate_f(fs, x + h / 2, y + k2 / 2)
        k4 = h * evaluate_f(fs, x + h, y + k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += h
        print(f"x = {x:.4f}, y = {y:.6f}")


# ---------------------------------------------------------------------------
# 18. FINITE DIFFERENCE METHOD FOR ELLIPTIC PDE (LAPLACE EQUATION)
# ---------------------------------------------------------------------------

def elliptic_pde():
    import numpy as np

    n = int(input("Enter number of intervals n: "))
    if n < 2:
        raise ValueError("n must be at least 2.")

    u = np.zeros((n + 1, n + 1))
    u[0] = list(map(float, input(f"Top boundary ({n + 1} values): ").split()))
    u[n] = list(map(float, input(f"Bottom boundary ({n + 1} values): ").split()))

    left = list(map(float, input(f"Left boundary, top to bottom ({n - 1} values): ").split()))
    right = list(map(float, input(f"Right boundary, top to bottom ({n - 1} values): ").split()))
    if len(left) != n - 1 or len(right) != n - 1:
        raise ValueError("Incorrect number of left/right boundary values.")

    u[1:n, 0] = left
    u[1:n, n] = right
    tol = float(input("Enter convergence tolerance (default suggestion 1e-6): ") or "1e-6")
    kmax = int(input("Enter maximum iterations (default suggestion 10000): ") or "10000")

    for iteration in range(1, kmax + 1):
        old = u.copy()
        for i in range(1, n):
            for j in range(1, n):
                u[i, j] = (
                    u[i - 1, j] + u[i + 1, j]
                    + u[i, j - 1] + u[i, j + 1]
                ) / 4.0
        error = np.max(np.abs(u - old))
        if error < tol:
            break

    print(f"\nSolution after {iteration} iterations:")
    print(np.round(u, 4))
    print(f"Maximum change = {error:.8g}")


# ---------------------------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------------------------

METHODS = [
    ("01. Gauss Elimination", gauss_elimination),
    ("02. Gauss Elimination with Row Pivoting", gauss_elimination_pivoting),
    ("03. Gauss-Jacobi", gauss_jacobi),
    ("04. Gauss-Seidel", gauss_seidel),
    ("05. Power Method", power_method),
    ("06. Gauss-Jordan Matrix Inverse", jordan_inverse),
    ("07. Lagrange Interpolation", lagrange_interpolation),
    ("08. Newton Divided Difference", newton_divided_difference),
    ("09. Newton Forward Difference", forward_difference),
    ("10. Newton Backward Difference", backward_difference),
    ("11. Natural Cubic Spline", cubic_spline),
    ("12. Trapezoidal Rule", trapezoidal_rule),
    ("13. Simpson's 1/3 Rule", simpsons_rule),
    ("14. Two-Point Gauss Quadrature", gauss_quadrature),
    ("15. Adams-Bashforth-Moulton", adams_bashforth_moulton),
    ("16. Milne Predictor-Corrector", milne_predictor_corrector),
    ("17. Runge-Kutta Fourth Order", runge_kutta),
    ("18. Elliptic PDE - Laplace Equation", elliptic_pde),
]


def main():
    while True:
        print("\n" + "=" * 64)
        print("S2.PY - NUMERICAL METHODS")
        print("=" * 64)
        for number, (title, _) in enumerate(METHODS, 1):
            print(f"{number:2}. {title}")
        print(" 0. Exit")

        choice = input("\nSelect a method: ").strip()
        if choice == "0":
            print("Exiting.")
            break
        if not choice.isdigit() or not 1 <= int(choice) <= len(METHODS):
            print("Invalid selection. Try again.")
            continue

        title, function = METHODS[int(choice) - 1]
        print(f"\n--- {title} ---")
        try:
            function()
        except (ValueError, IndexError, ZeroDivisionError) as error:
            print(f"Error: {error}")
        except KeyboardInterrupt:
            print("\nMethod cancelled.")
        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()
