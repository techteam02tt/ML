"""
NUMERICAL METHODS - PYTHON REFERENCE
====================================

A single-file reference for common numerical methods.

Requirements:
    pip install sympy matplotlib numpy

Run:
    python numerical_methods_reference.py

Choose a method from the menu. Each method runs independently.

Note:
    This file is intended for learning/reference. The expression-based ODE
    examples use eval() for convenience; do not run untrusted expressions.
"""

import math
import sys


# ---------------------------------------------------------------------------
# 01. GAUSS ELIMINATION
# ---------------------------------------------------------------------------

def gauss_elimination_basic():
    n = int(input("Enter the size of the system: "))
    print("Enter coefficient matrix A, one row per line:")
    A = [list(map(float, input().split())) for _ in range(n)]
    print("Enter RHS vector b:")
    b = list(map(float, input().split()))

    for k in range(n - 1):
        if abs(A[k][k]) < 1e-12:
            raise ValueError("Zero pivot. Use Gauss elimination with row pivoting.")
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    if abs(A[n - 1][n - 1]) < 1e-12:
        raise ValueError("No unique solution exists.")

    x = [0.0] * n
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        total = b[i]
        for j in range(i + 1, n):
            total -= A[i][j] * x[j]
        if abs(A[i][i]) < 1e-12:
            raise ValueError("Zero pivot during back substitution.")
        x[i] = total / A[i][i]

    print("\nSolution:")
    for i, value in enumerate(x, 1):
        print(f"x{i} = {value:.6f}")


# ---------------------------------------------------------------------------
# 02. GAUSS ELIMINATION WITH PARTIAL ROW PIVOTING
# ---------------------------------------------------------------------------

def gauss_elimination_pivoting():
    n = int(input("Enter the size of the system: "))
    print("Enter coefficient matrix A, one row per line:")
    A = [list(map(float, input().split())) for _ in range(n)]
    print("Enter RHS vector b:")
    b = list(map(float, input().split()))
    tol = 1e-12

    def print_matrix():
        for row, rhs in zip(A, b):
            print("[" + " ".join(f"{v:10.5f}" for v in row) + f" | {rhs:10.5f}]")

    for k in range(n - 1):
        max_row = max(range(k, n), key=lambda r: abs(A[r][k]))
        if abs(A[max_row][k]) <= tol:
            raise ValueError("No unique solution exists.")
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]
        print(f"\nAfter pivoting at column {k + 1}:")
        print_matrix()

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]
            A[i][k] = 0.0
        print("After elimination:")
        print_matrix()

    if abs(A[n - 1][n - 1]) <= tol:
        raise ValueError("No unique solution exists.")

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) <= tol:
            raise ValueError("No unique solution exists.")
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]

    print("\nSolution:")
    for i, value in enumerate(x, 1):
        print(f"x{i} = {value:.6f}")


# ---------------------------------------------------------------------------
# 03. GAUSS-JACOBI ITERATIVE METHOD
# ---------------------------------------------------------------------------

def gauss_jacobi():
    n = int(input("Enter number of equations: "))
    print("Enter coefficient matrix A, one row per line:")
    A = [list(map(float, input().split())) for _ in range(n)]
    b = list(map(float, input("Enter RHS vector b: ").split()))
    x_old = list(map(float, input("Enter initial guess values: ").split()))
    tol = float(input("Enter tolerance: "))
    kmax = int(input("Enter maximum iterations: "))

    if any(abs(A[i][i]) < 1e-12 for i in range(n)):
        raise ValueError("Diagonal elements must be non-zero.")

    for k in range(1, kmax + 1):
        x_new = [0.0] * n
        for i in range(n):
            total = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - total) / A[i][i]
        error = max(abs(x_new[i] - x_old[i]) for i in range(n))
        print(f"Iteration {k}: {x_new}")
        print(f"Error: {error:.8g}")
        x_old = x_new
        if error <= tol:
            break

    print("\nFinal solution:")
    for i, value in enumerate(x_old, 1):
        print(f"x{i} = {value:.8f}")


# ---------------------------------------------------------------------------
# 04. GAUSS-SEIDEL ITERATIVE METHOD
# ---------------------------------------------------------------------------

def gauss_seidel():
    n = int(input("Enter number of equations: "))
    print("Enter coefficient matrix A, one row per line:")
    A = [list(map(float, input().split())) for _ in range(n)]
    b = list(map(float, input("Enter RHS vector b: ").split()))
    x = list(map(float, input("Enter initial guess values: ").split()))
    tol = float(input("Enter tolerance: "))
    kmax = int(input("Enter maximum iterations: "))

    if any(abs(A[i][i]) < 1e-12 for i in range(n)):
        raise ValueError("Diagonal elements must be non-zero.")

    for k in range(1, kmax + 1):
        old = x.copy()
        for i in range(n):
            total = sum(A[i][j] * x[j] for j in range(i))
            total += sum(A[i][j] * old[j] for j in range(i + 1, n))
            x[i] = (b[i] - total) / A[i][i]
        error = max(abs(x[i] - old[i]) for i in range(n))
        print(f"Iteration {k}: {x}")
        print(f"Error: {error:.8g}")
        if error <= tol:
            break

    print("\nFinal solution:")
    for i, value in enumerate(x, 1):
        print(f"x{i} = {value:.8f}")


# ---------------------------------------------------------------------------
# 05. POWER METHOD FOR DOMINANT EIGENVALUE
# ---------------------------------------------------------------------------

def power_method():
    n = int(input("Enter order of matrix: "))
    print("Enter matrix A, one row per line:")
    A = [list(map(float, input().split())) for _ in range(n)]
    v = list(map(float, input("Enter initial vector: ").split()))
    tol = float(input("Enter tolerance: "))
    kmax = int(input("Enter maximum iterations: "))

    if len(v) != n or max(map(abs, v)) == 0:
        raise ValueError("Initial vector must be non-zero and have length n.")

    eigenvalue_old = None
    for k in range(1, kmax + 1):
        w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        scale = max(map(abs, w))
        if scale < 1e-14:
            raise ValueError("Power iteration reached a zero vector.")
        v = [value / scale for value in w]
        # Rayleigh quotient gives a useful eigenvalue estimate.
        Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        denom = sum(value * value for value in v)
        eigenvalue = sum(v[i] * Av[i] for i in range(n)) / denom
        error = (abs(eigenvalue - eigenvalue_old)
                 if eigenvalue_old is not None else float("inf"))
        print(f"Iteration {k}: eigenvalue = {eigenvalue:.8f}, vector = {v}")
        if eigenvalue_old is not None and error <= tol:
            break
        eigenvalue_old = eigenvalue

    print(f"\nDominant eigenvalue estimate = {eigenvalue:.8f}")
    print("Corresponding eigenvector estimate =", v)


# ---------------------------------------------------------------------------
# 06. GAUSS-JORDAN METHOD FOR MATRIX INVERSE
# ---------------------------------------------------------------------------

def jordan_inverse():
    n = int(input("Enter size of square matrix: "))
    print("Enter matrix A, one row per line:")
    A = [list(map(float, input().split())) for _ in range(n)]
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]
    tol = 1e-12

    print("\nInitial augmented matrix [A | I]:")
    for row in aug:
        print(" ".join(f"{v:9.3f}" for v in row))

    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot_row][i]) < tol:
            raise ValueError("Matrix is singular; inverse does not exist.")
        aug[i], aug[pivot_row] = aug[pivot_row], aug[i]
        pivot = aug[i][i]
        aug[i] = [value / pivot for value in aug[i]]

        for k in range(n):
            if k != i:
                factor = aug[k][i]
                aug[k] = [aug[k][j] - factor * aug[i][j]
                          for j in range(2 * n)]

        print(f"\nIteration {i + 1}:")
        for row in aug:
            print(" ".join(f"{v:9.3f}" for v in row))

    inverse = [row[n:] for row in aug]
    print("\nInverse matrix:")
    for row in inverse:
        print(" ".join(f"{v:.6f}" for v in row))

    print("\nA * A-inverse (should be identity):")
    for i in range(n):
        row = [sum(A[i][k] * inverse[k][j] for k in range(n))
               for j in range(n)]
        print(" ".join(f"{v:.6f}" for v in row))


# ---------------------------------------------------------------------------
# 07. LAGRANGE INTERPOLATION
# ---------------------------------------------------------------------------

def lagrange_interpolation():
    import sympy as sp

    n = int(input("Enter number of data points: "))
    points = [list(map(float, input(f"Enter x and y for point {i + 1}: ").split()))
              for i in range(n)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(set(xs)) != n:
        raise ValueError("All x values must be distinct.")
    x_value = float(input("Enter x value to interpolate: "))
    x = sp.Symbol("x")
    polynomial = 0
    value = 0.0

    for i in range(n):
        basis = 1
        basis_at_value = 1.0
        for j in range(n):
            if i != j:
                basis *= (x - xs[j]) / (xs[i] - xs[j])
                basis_at_value *= (x_value - xs[j]) / (xs[i] - xs[j])
        polynomial += ys[i] * basis
        value += ys[i] * basis_at_value

    polynomial = sp.expand(polynomial)
    print("\nLagrange polynomial:", polynomial)
    print(f"Interpolated value at x = {x_value}: {value:.8f}")
    print("Polynomial degree:", sp.degree(polynomial, x))
    print("Symbolic operation count:", sp.count_ops(polynomial))


# ---------------------------------------------------------------------------
# 08. NEWTON DIVIDED DIFFERENCE INTERPOLATION
# ---------------------------------------------------------------------------

def newton_divided_difference():
    n = int(input("Enter number of data points: "))
    x, y = [], []
    print("Enter each x and y value:")
    for i in range(n):
        xi, yi = map(float, input(f"Point {i + 1}: ").split())
        x.append(xi)
        y.append(yi)
    if len(set(x)) != n:
        raise ValueError("All x values must be distinct.")

    d = [[0.0] * n for _ in range(n)]
    for j in range(n):
        d[0][j] = y[j]
    for i in range(1, n):
        for j in range(n - i):
            d[i][j] = (d[i - 1][j + 1] - d[i - 1][j]) / (x[j + i] - x[j])

    print("\nDivided difference table:")
    print(f"{'x':<12}{'y':<12}" + "".join(f"{'dd' + str(i):<12}" for i in range(1, n)))
    for j in range(n):
        row = f"{x[j]:<12.4f}{y[j]:<12.4f}"
        for i in range(1, n - j):
            row += f"{d[i][j]:<12.4f}"
        print(row)

    equation = f"{d[0][0]:.8g}"
    for i in range(1, n):
        term = f"({d[i][0]:.8g})"
        for j in range(i):
            term += f"(x - ({x[j]:.8g}))"
        equation += " + " + term
    print("\nNewton divided-difference form:")
    print("P(x) =", equation)

    x_value = float(input("Enter x value to interpolate: "))
    result = d[0][0]
    product = 1.0
    for i in range(1, n):
        product *= (x_value - x[i - 1])
        result += d[i][0] * product
    print(f"Interpolated value at x = {x_value}: {result:.8f}")


# ---------------------------------------------------------------------------
# 09. NEWTON FORWARD DIFFERENCE INTERPOLATION
# ---------------------------------------------------------------------------

def newton_forward_difference():
    n = int(input("Enter number of data points: "))
    x, y = [], []
    for i in range(n):
        xi, yi = map(float, input(f"Enter x[{i}] and y[{i}]: ").split())
        x.append(xi)
        y.append(yi)
    if n < 2:
        raise ValueError("At least two points are required.")
    h = x[1] - x[0]
    if abs(h) < 1e-12 or any(abs((x[i] - x[i - 1]) - h) > 1e-9 for i in range(2, n)):
        raise ValueError("x values must be equally spaced.")

    diff = [y.copy()]
    for order in range(1, n):
        previous = diff[-1]
        diff.append([previous[j + 1] - previous[j] for j in range(len(previous) - 1)])

    print("\nForward difference table:")
    for i in range(n):
        row = [f"{x[i]:.4f}", f"{y[i]:.4f}"]
        row += [f"{diff[k][i]:.4f}" if i < len(diff[k]) else "" for k in range(1, n)]
        print("\t".join(row))

    print("\nNewton forward formula:")
    print("P(x) = y0 + u Δy0 + u(u-1) Δ²y0/2! + ...")
    count = int(input("How many x values do you want to interpolate? "))
    for _ in range(count):
        x_value = float(input("Enter x value: "))
        u = (x_value - x[0]) / h
        result = y[0]
        product = 1.0
        factorial = 1
        for order in range(1, n):
            product *= (u - (order - 1))
            factorial *= order
            result += product * diff[order][0] / factorial
        print(f"u = ({x_value} - {x[0]}) / {h} = {u}")
        print(f"P({x_value}) = {result:.8f}")


# ---------------------------------------------------------------------------
# 10. NEWTON BACKWARD DIFFERENCE INTERPOLATION
# ---------------------------------------------------------------------------

def newton_backward_difference():
    import matplotlib.pyplot as plt

    n = int(input("Enter number of data points: "))
    x, y = [], []
    for i in range(n):
        xi, yi = map(float, input(f"Enter x[{i}] and y[{i}]: ").split())
        x.append(xi)
        y.append(yi)
    if n < 2:
        raise ValueError("At least two points are required.")
    h = x[1] - x[0]
    if abs(h) < 1e-12 or any(abs((x[i] - x[i - 1]) - h) > 1e-9 for i in range(2, n)):
        raise ValueError("x values must be equally spaced.")

    diff = [y.copy()]
    for order in range(1, n):
        previous = diff[-1]
        diff.append([previous[j + 1] - previous[j] for j in range(len(previous) - 1)])

    print("\nBackward difference table:")
    for i in range(n):
        row = [f"{x[i]:.4f}", f"{y[i]:.4f}"]
        row += [f"{diff[k][i-k]:.4f}" if i >= k else "" for k in range(1, n)]
        print("\t".join(row))

    print("\nNewton backward formula:")
    print("P(x) = yn + u ∇yn + u(u+1) ∇²yn/2! + ...")
    count = int(input("How many x values do you want to interpolate? "))
    x_inter, y_inter = [], []
    for _ in range(count):
        x_value = float(input("Enter x value: "))
        u = (x_value - x[-1]) / h
        result = y[-1]
        product = 1.0
        factorial = 1
        for order in range(1, n):
            product *= (u + order - 1)
            factorial *= order
            result += product * diff[order][-1] / factorial
        x_inter.append(x_value)
        y_inter.append(result)
        print(f"u = ({x_value} - {x[-1]}) / {h} = {u}")
        print(f"P({x_value}) = {result:.8f}")

    plt.plot(x, y, "bo-", label="Given data")
    if x_inter:
        plt.plot(x_inter, y_inter, "go", label="Interpolated points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Newton Backward Interpolation")
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# 11. NATURAL CUBIC SPLINE INTERPOLATION
# ---------------------------------------------------------------------------

def cubic_spline():
    import sympy as sp
    import matplotlib.pyplot as plt

    n = int(input("Enter number of data points: "))
    if n < 3:
        raise ValueError("At least three data points are required.")
    x_data, y_data = [], []
    for i in range(n):
        xv, yv = map(float, input(f"Enter x[{i}] and y[{i}]: ").split())
        x_data.append(xv)
        y_data.append(yv)
    if any(x_data[i] <= x_data[i - 1] for i in range(1, n)):
        raise ValueError("x values must be strictly increasing.")

    x = sp.Symbol("x")
    h = [x_data[i] - x_data[i - 1] for i in range(1, n)]
    # Natural spline: second derivatives at the endpoints are zero.
    m = [0.0] * n
    if n > 2:
        matrix = sp.zeros(n - 2, n - 2)
        rhs = sp.zeros(n - 2, 1)
        for i in range(1, n - 1):
            row = i - 1
            if i - 1 >= 1:
                matrix[row, row - 1] = h[i - 1]
            matrix[row, row] = 2 * (h[i - 1] + h[i])
            if i + 1 <= n - 2:
                matrix[row, row + 1] = h[i]
            rhs[row] = 6 * ((y_data[i + 1] - y_data[i]) / h[i]
                            - (y_data[i] - y_data[i - 1]) / h[i - 1])
        solution = matrix.LUsolve(rhs)
        for i in range(1, n - 1):
            m[i] = solution[i - 1]

    functions = []
    for i in range(1, n):
        hi = h[i - 1]
        xi0, xi1 = x_data[i - 1], x_data[i]
        expression = (
            m[i - 1] * (xi1 - x) ** 3 / (6 * hi)
            + m[i] * (x - xi0) ** 3 / (6 * hi)
            + (y_data[i - 1] - m[i - 1] * hi ** 2 / 6) * (xi1 - x) / hi
            + (y_data[i] - m[i] * hi ** 2 / 6) * (x - xi0) / hi
        )
        functions.append(sp.expand(expression))

    print("\nNatural cubic spline functions:")
    for i, function in enumerate(functions, 1):
        print(f"S{i}(x) =", function)

    count = int(input("Enter number of interpolation values: "))
    for _ in range(count):
        xv = float(input("Enter interpolation x value: "))
        if not x_data[0] <= xv <= x_data[-1]:
            print("Value lies outside the data interval.")
            continue
        interval = next((i for i in range(n - 1)
                         if x_data[i] <= xv <= x_data[i + 1]), n - 2)
        print(f"S({xv}) = {float(functions[interval].subs(x, xv)):.8f}")

    for i, function in enumerate(functions):
        plot = sp.plot(function, (x, x_data[i], x_data[i + 1]),
                       show=False, label=f"S{i + 1}(x)")
        if i == 0:
            final_plot = plot
        else:
            final_plot.extend(plot)
    final_plot.show()
    plt.plot(x_data, y_data, "ro")
    plt.title("Cubic Spline Data Points")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# 12. COMPOSITE TRAPEZOIDAL RULE
# ---------------------------------------------------------------------------

def trapezoidal_rule():
    import sympy as sp

    n = int(input("Enter number of subintervals: "))
    if n < 1:
        raise ValueError("Number of subintervals must be positive.")
    a = float(input("Enter lower limit a: "))
    b = float(input("Enter upper limit b: "))
    expr = sp.sympify(input("Enter f(x), e.g. x**2: "))
    x = sp.Symbol("x")
    f = sp.lambdify(x, expr, "math")
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += 2 * f(a + i * h)
    approx = h * total / 2
    exact = float(sp.integrate(expr, (x, a, b)))
    print("\nf(x) =", expr)
    print("Step size h =", h)
    print("Approximate integral =", approx)
    print("Exact integral =", exact)
    print("Absolute error =", abs(exact - approx))


# ---------------------------------------------------------------------------
# 13. COMPOSITE SIMPSON'S 1/3 RULE
# ---------------------------------------------------------------------------

def simpsons_one_third_rule():
    import sympy as sp

    n = int(input("Enter number of subintervals (must be even): "))
    if n < 2 or n % 2:
        raise ValueError("Simpson's 1/3 rule requires a positive even number of subintervals.")
    a = float(input("Enter lower limit a: "))
    b = float(input("Enter upper limit b: "))
    expr = sp.sympify(input("Enter f(x), e.g. x**2: "))
    x = sp.Symbol("x")
    f = sp.lambdify(x, expr, "math")
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    approx = h * total / 3
    exact = float(sp.integrate(expr, (x, a, b)))
    print("\nComposite Simpson's 1/3 approximation =", approx)
    print("Exact integral =", exact)
    print("Absolute error =", abs(exact - approx))


# ---------------------------------------------------------------------------
# 14. COMPOSITE TWO-POINT GAUSS-LEGENDRE QUADRATURE
# ---------------------------------------------------------------------------

def gauss_quadrature_two_point():
    import sympy as sp

    n = int(input("Enter number of subintervals: "))
    if n < 1:
        raise ValueError("Number of subintervals must be positive.")
    a = float(input("Enter lower limit a: "))
    b = float(input("Enter upper limit b: "))
    expr = sp.sympify(input("Enter f(x), e.g. x**2: "))
    x = sp.Symbol("x")
    f = sp.lambdify(x, expr, "math")
    h = (b - a) / n
    offset_factor = 1 / math.sqrt(3)
    approx = 0.0

    for i in range(n):
        left = a + i * h
        right = left + h
        midpoint = (left + right) / 2
        half_width = h / 2
        x1 = midpoint - half_width * offset_factor
        x2 = midpoint + half_width * offset_factor
        approx += half_width * (f(x1) + f(x2))

    exact = float(sp.integrate(expr, (x, a, b)))
    print("\nComposite two-point Gauss quadrature =", approx)
    print("Exact integral =", exact)
    print("Absolute error =", abs(exact - approx))


# ---------------------------------------------------------------------------
# COMMON HELPER FOR ODE METHODS
# ---------------------------------------------------------------------------

def evaluate_ode_rhs(expression, x, y):
    """Evaluate a user-entered RHS such as 'x + y'."""
    allowed = {
        "x": x, "y": y, "math": math, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "exp": math.exp, "sqrt": math.sqrt, "log": math.log,
        "pi": math.pi, "e": math.e, "abs": abs
    }
    return float(eval(expression, {"__builtins__": {}}, allowed))


# ---------------------------------------------------------------------------
# 15. ADAMS-BASHFORTH-MOULTON PREDICTOR-CORRECTOR (4-STEP)
# ---------------------------------------------------------------------------

def adams_bashforth_moulton():
    import matplotlib.pyplot as plt

    expression = input("Enter f(x, y), e.g. x + y: ")
    x, y = [], []
    print("Enter 4 starting values as x y pairs, one pair per line:")
    for _ in range(4):
        xv, yv = map(float, input().split())
        x.append(xv)
        y.append(yv)
    target = float(input("Enter x value at which y is required: "))
    h = x[1] - x[0]
    if abs(h) < 1e-12 or any(abs((x[i] - x[i - 1]) - h) > 1e-8 for i in range(1, 4)):
        raise ValueError("Starting x values must be equally spaced.")
    steps_float = (target - x[-1]) / h
    steps = round(steps_float)
    if steps < 0 or abs(steps_float - steps) > 1e-8:
        raise ValueError("Target must be ahead of the last x value by a whole number of steps.")

    print("\nAdams-Bashforth-Moulton results:")
    for _ in range(steps):
        i = len(x) - 1
        F = [evaluate_ode_rhs(expression, x[j], y[j]) for j in range(i - 3, i + 1)]
        x_next = x[i] + h
        y_pred = y[i] + h / 24 * (55 * F[3] - 59 * F[2] + 37 * F[1] - 9 * F[0])
        for iteration in range(1, 101):
            f_pred = evaluate_ode_rhs(expression, x_next, y_pred)
            y_corr = y[i] + h / 24 * (9 * f_pred + 19 * F[3] - 5 * F[2] + F[1])
            if abs(y_corr - y_pred) < 1e-6:
                break
            y_pred = y_corr
        x.append(x_next)
        y.append(y_corr)
        print(f"x = {x_next:.6f}, predicted/corrected y = {y_corr:.6f}, corrections = {iteration}")

    print(f"\nApproximate y({target}) = {y[-1]:.8f}")
    plt.plot(x, y, "o-")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Adams-Bashforth-Moulton Method")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# 16. MILNE'S PREDICTOR-CORRECTOR METHOD
# ---------------------------------------------------------------------------

def milne_predictor_corrector():
    import matplotlib.pyplot as plt

    expression = input("Enter f(x, y), e.g. x + y: ")
    x, y = [], []
    print("Enter 4 starting values as x y pairs, one pair per line:")
    for _ in range(4):
        xv, yv = map(float, input().split())
        x.append(xv)
        y.append(yv)
    target = float(input("Enter x value at which y is required: "))
    h = x[1] - x[0]
    if abs(h) < 1e-12 or any(abs((x[i] - x[i - 1]) - h) > 1e-8 for i in range(1, 4)):
        raise ValueError("Starting x values must be equally spaced.")
    steps_float = (target - x[-1]) / h
    steps = round(steps_float)
    if steps < 0 or abs(steps_float - steps) > 1e-8:
        raise ValueError("Target must be ahead of the last x value by a whole number of steps.")

    print("\nMilne predictor-corrector results:")
    for _ in range(steps):
        i = len(x) - 1
        F = [evaluate_ode_rhs(expression, x[j], y[j]) for j in range(i - 3, i + 1)]
        x_next = x[i] + h
        # Milne predictor: y_(n+1) = y_(n-3) + 4h/3 * (2f_(n-2) - f_(n-1) + 2f_n)
        y_pred = y[i - 3] + (4 * h / 3) * (2 * F[1] - F[2] + 2 * F[3])
        for iteration in range(1, 101):
            f_pred = evaluate_ode_rhs(expression, x_next, y_pred)
            y_corr = y[i - 1] + h / 3 * (F[2] + 4 * F[3] + f_pred)
            if abs(y_corr - y_pred) < 1e-6:
                break
            y_pred = y_corr
        x.append(x_next)
        y.append(y_corr)
        print(f"x = {x_next:.6f}, predicted/corrected y = {y_corr:.6f}, corrections = {iteration}")

    print(f"\nApproximate y({target}) = {y[-1]:.8f}")
    plt.plot(x, y, "o-")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Milne Predictor-Corrector Method")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# 17. RUNGE-KUTTA FOURTH-ORDER METHOD
# ---------------------------------------------------------------------------

def runge_kutta_fourth_order():
    import matplotlib.pyplot as plt

    expression = input("Enter f(x, y), e.g. x + y: ")
    x = float(input("Enter initial x (x0): "))
    y = float(input("Enter initial y (y0): "))
    h = float(input("Enter step size h: "))
    target = float(input("Enter target x value: "))
    if h == 0:
        raise ValueError("Step size cannot be zero.")
    steps_float = (target - x) / h
    steps = round(steps_float)
    if steps < 0 or abs(steps_float - steps) > 1e-8:
        raise ValueError("Target must be reachable by an integer number of steps.")

    x_values, y_values = [x], [y]
    print("\nRunge-Kutta fourth-order results:")
    for _ in range(steps):
        k1 = h * evaluate_ode_rhs(expression, x, y)
        k2 = h * evaluate_ode_rhs(expression, x + h / 2, y + k1 / 2)
        k3 = h * evaluate_ode_rhs(expression, x + h / 2, y + k2 / 2)
        k4 = h * evaluate_ode_rhs(expression, x + h, y + k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += h
        x_values.append(x)
        y_values.append(y)
        print(f"x = {x:.6f}, y = {y:.8f}")

    plt.plot(x_values, y_values, "o-")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Runge-Kutta Fourth-Order Method")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# 18A. FINITE DIFFERENCE METHOD FOR ELLIPTIC PDE (LAPLACE EQUATION)
#      Liebmann / Gauss-Seidel method for a rectangular grid
# ---------------------------------------------------------------------------

def elliptic_pde_liebmann():
    print("Elliptic PDE: Laplace equation using Liebmann (Gauss-Seidel) method")
    nx = int(input("Enter number of x intervals (nx): "))
    ny = int(input("Enter number of y intervals (ny): "))
    if nx < 2 or ny < 2:
        raise ValueError("At least 2 intervals in each direction are required.")
    top = float(input("Top boundary value: "))
    bottom = float(input("Bottom boundary value: "))
    left = float(input("Left boundary value: "))
    right = float(input("Right boundary value: "))
    tol = float(input("Enter convergence tolerance: "))
    max_iter = int(input("Enter maximum iterations: "))

    grid = [[0.0 for _ in range(nx + 1)] for _ in range(ny + 1)]
    for i in range(nx + 1):
        grid[0][i] = bottom
        grid[ny][i] = top
    for j in range(ny + 1):
        grid[j][0] = left
        grid[j][nx] = right
    grid[0][0] = (bottom + left) / 2
    grid[0][nx] = (bottom + right) / 2
    grid[ny][0] = (top + left) / 2
    grid[ny][nx] = (top + right) / 2

    avg = (top + bottom + left + right) / 4
    for j in range(1, ny):
        for i in range(1, nx):
            grid[j][i] = avg

    for iteration in range(1, max_iter + 1):
        max_error = 0.0
        for j in range(ny - 1, 0, -1):
            for i in range(1, nx):
                old = grid[j][i]
                grid[j][i] = 0.25 * (
                    grid[j + 1][i] + grid[j - 1][i]
                    + grid[j][i + 1] + grid[j][i - 1]
                )
                max_error = max(max_error, abs(grid[j][i] - old))
        if iteration == 1 or iteration % 10 == 0:
            print(f"Iteration {iteration}: max change = {max_error:.8g}")
        if max_error < tol:
            print(f"Converged in {iteration} iterations.")
            break
    else:
        print("Maximum iterations reached without meeting tolerance.")

    print("\nFinal grid (top row to bottom row):")
    for row in reversed(grid):
        print(" ".join(f"{value:10.4f}" for value in row))


# ---------------------------------------------------------------------------
# 18B. FINITE DIFFERENCE METHOD FOR ELLIPTIC PDE (GENERAL RECTANGLE MESH)
#      Laplace equation; Gauss-Seidel method. For h != k use weighted stencil.
# ---------------------------------------------------------------------------

def elliptic_pde_rectangular_mesh():
    N = int(input("Enter number of x intervals (N): "))
    M = int(input("Enter number of y intervals (M): "))
    tol = float(input("Enter tolerance: "))
    max_iter = int(input("Enter maximum iterations: "))
    a = float(input("Enter starting x (a): "))
    b = float(input("Enter ending x (b): "))
    c = float(input("Enter starting y (c): "))
    d = float(input("Enter ending y (d): "))
    if N < 2 or M < 2 or b <= a or d <= c:
        raise ValueError("Use at least 2 intervals and increasing domain limits.")

    h = (b - a) / N
    k = (d - c) / M
    print(f"\nh = {h}, k = {k}")
    bottom = float(input("Enter bottom boundary value: "))
    top = float(input("Enter top boundary value: "))
    left = float(input("Enter left boundary value: "))
    right = float(input("Enter right boundary value: "))

    U = [[0.0 for _ in range(N + 1)] for _ in range(M + 1)]
    for i in range(N + 1):
        U[0][i] = bottom
        U[M][i] = top
    for j in range(M + 1):
        U[j][0] = left
        U[j][N] = right
    U[0][0] = (bottom + left) / 2
    U[0][N] = (bottom + right) / 2
    U[M][0] = (top + left) / 2
    U[M][N] = (top + right) / 2

    # Discrete Laplace stencil for unequal mesh sizes.
    wx = 1 / h**2
    wy = 1 / k**2
    denominator = 2 * (wx + wy)
    for iteration in range(1, max_iter + 1):
        old = [row.copy() for row in U]
        for j in range(1, M):
            for i in range(1, N):
                U[j][i] = (
                    wx * (U[j][i - 1] + U[j][i + 1])
                    + wy * (U[j - 1][i] + U[j + 1][i])
                ) / denominator
        error = max(abs(U[j][i] - old[j][i])
                    for j in range(M + 1) for i in range(N + 1))
        print(f"Iteration {iteration}: max change = {error:.8g}")
        if error < tol:
            print(f"Converged in {iteration} iterations.")
            break
    else:
        print("Maximum iterations reached without meeting tolerance.")

    print("\nFinal approximate solution:")
    for row in reversed(U):
        print(" ".join(f"{value:10.6f}" for value in row))


# ---------------------------------------------------------------------------
# 19A. PARABOLIC PDE: 1D HEAT EQUATION (BENDER-SCHMIDT / FTCS)
# ---------------------------------------------------------------------------

def heat_equation_ftcs_custom():
    print("1D Heat Equation: explicit FTCS / Bender-Schmidt scheme")
    length = float(input("Enter rod length L: "))
    alpha = float(input("Enter thermal diffusivity alpha (c^2): "))
    nx = int(input("Enter number of spatial intervals nx: "))
    nt = int(input("Enter number of time steps nt: "))
    total_time = float(input("Enter total time T: "))
    if length <= 0 or alpha <= 0 or nx < 2 or nt < 1 or total_time <= 0:
        raise ValueError("Invalid dimensions, diffusivity, or time settings.")

    h = length / nx
    dt = total_time / nt
    lam = alpha * dt / h**2
    print(f"h = {h:.6f}, dt = {dt:.6f}, lambda = {lam:.6f}")
    if lam > 0.5 + 1e-12:
        print("WARNING: Explicit scheme is unstable for lambda > 0.5.")

    expr = input("Enter initial condition f(x), or press Enter for sin(pi*x/L): ").strip()
    if expr:
        def initial(x):
            allowed = {"x": x, "math": math, "sin": math.sin, "cos": math.cos,
                       "exp": math.exp, "pi": math.pi}
            return float(eval(expr, {"__builtins__": {}}, allowed))
    else:
        initial = lambda x: math.sin(math.pi * x / length)

    left = float(input("Boundary value at x=0: "))
    right = float(input("Boundary value at x=L: "))
    u = [[0.0] * (nx + 1) for _ in range(nt + 1)]
    for i in range(nx + 1):
        u[0][i] = initial(i * h)
    for j in range(nt + 1):
        u[j][0] = left
        u[j][nx] = right

    for j in range(nt):
        for i in range(1, nx):
            u[j + 1][i] = (
                lam * u[j][i - 1] + (1 - 2 * lam) * u[j][i]
                + lam * u[j][i + 1]
            )

    print("\nTemperature distribution u(x,t):")
    print("Time\t" + "\t".join(f"x={i*h:.3f}" for i in range(nx + 1)))
    for j in range(nt + 1):
        print(f"{j*dt:.4f}\t" + "\t".join(f"{value:.5f}" for value in u[j]))


# ---------------------------------------------------------------------------
# 19B. PARABOLIC PDE: FTCS HEAT EQUATION WITH SINE INITIAL CONDITION
# ---------------------------------------------------------------------------

def heat_equation_ftcs_sine():
    length = float(input("Enter rod length L: "))
    total_time = float(input("Enter total time T: "))
    alpha = float(input("Enter thermal diffusivity alpha: "))
    M = int(input("Enter spatial intervals M: "))
    N = int(input("Enter time steps N: "))
    if length <= 0 or total_time <= 0 or alpha <= 0 or M < 2 or N < 1:
        raise ValueError("Invalid input.")

    h = length / M
    dt = total_time / N
    r = alpha * dt / h**2
    print(f"h = {h}, dt = {dt}, r = {r}")
    if r > 0.5:
        print("WARNING: Stability condition r <= 0.5 is violated.")
    else:
        print("Stability condition satisfied.")

    x = [i * h for i in range(M + 1)]
    U = [100 * math.sin(math.pi * xv / length) for xv in x]
    U[0] = U[M] = 0.0
    print("\nInitial temperature:", U)

    for step in range(N):
        U_new = U.copy()
        for i in range(1, M):
            U_new[i] = r * U[i - 1] + (1 - 2 * r) * U[i] + r * U[i + 1]
        U_new[0] = U_new[M] = 0.0
        U = U_new
        print(f"\nTime step {step + 1}, t = {(step + 1) * dt:.4f}")
        for xv, value in zip(x, U):
            print(f"U({xv:.3f}) = {value:.6f}")

    print("\nFinal solution:")
    for xv, value in zip(x, U):
        print(f"U({xv:.3f}) = {value:.6f}")


# ---------------------------------------------------------------------------
# 19C. PARABOLIC PDE: CRANK-NICOLSON METHOD
# ---------------------------------------------------------------------------

def crank_nicolson_heat_equation():
    import numpy as np

    def thomas_algorithm(lower, diagonal, upper, rhs):
        n = len(rhs)
        c_prime = [0.0] * n
        d_prime = [0.0] * n
        if abs(diagonal[0]) < 1e-14:
            raise ValueError("Zero pivot in Thomas algorithm.")
        c_prime[0] = upper[0] / diagonal[0] if n > 1 else 0.0
        d_prime[0] = rhs[0] / diagonal[0]
        for i in range(1, n):
            denom = diagonal[i] - lower[i] * c_prime[i - 1]
            if abs(denom) < 1e-14:
                raise ValueError("Zero pivot in Thomas algorithm.")
            c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
            d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom
        solution = [0.0] * n
        solution[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
        return solution

    print("Crank-Nicolson method for 1D heat equation")
    length = float(input("Enter rod length L: "))
    total_time = float(input("Enter total time T: "))
    alpha = float(input("Enter thermal diffusivity alpha: "))
    M = int(input("Enter spatial intervals M: "))
    N = int(input("Enter time steps N: "))
    if length <= 0 or total_time <= 0 or alpha <= 0 or M < 2 or N < 1:
        raise ValueError("Invalid input.")

    h = length / M
    dt = total_time / N
    r = alpha * dt / h**2
    x = [i * h for i in range(M + 1)]
    U = [100 * math.sin(math.pi * xv / length) for xv in x]
    U[0] = U[M] = 0.0
    p = M - 1
    lower = [-r / 2] * p
    diagonal = [1 + r] * p
    upper = [-r / 2] * p

    print(f"h = {h}, dt = {dt}, r = {r}")
    print("Initial temperature:", U)
    for step in range(N):
        rhs = [0.0] * p
        for j in range(p):
            i = j + 1
            rhs[j] = (r / 2) * U[i - 1] + (1 - r) * U[i] + (r / 2) * U[i + 1]
        interior = thomas_algorithm(lower, diagonal, upper, rhs)
        U = [0.0] + interior + [0.0]
        print(f"\nTime step {step + 1}, t = {(step + 1) * dt:.4f}")
        for xv, value in zip(x, U):
            print(f"U({xv:.3f}) = {value:.6f}")

    print("\nFinal temperature distribution:", U)


# ---------------------------------------------------------------------------
# 20A. HYPERBOLIC PDE: 1D WAVE EQUATION (EXPLICIT FINITE DIFFERENCE)
# ---------------------------------------------------------------------------

def wave_equation_finite_difference():
    length = float(input("Enter string length L: "))
    speed = float(input("Enter wave speed c: "))
    nx = int(input("Enter number of spatial divisions nx: "))
    nt = int(input("Enter number of time steps nt: "))
    total_time = float(input("Enter total time T: "))
    if length <= 0 or speed <= 0 or nx < 2 or nt < 1 or total_time <= 0:
        raise ValueError("Invalid input.")

    h = length / nx
    dt = total_time / nt
    lam = speed * dt / h
    lam2 = lam**2
    print(f"h = {h}, dt = {dt}, Courant number lambda = {lam}")
    if lam > 1 + 1e-12:
        print("WARNING: CFL stability condition lambda <= 1 is violated.")

    expr = input("Enter initial displacement f(x), or Enter for sin(pi*x/L): ").strip()
    if expr:
        def displacement(x):
            allowed = {"x": x, "math": math, "sin": math.sin, "cos": math.cos,
                       "exp": math.exp, "pi": math.pi}
            return float(eval(expr, {"__builtins__": {}}, allowed))
    else:
        displacement = lambda x: math.sin(math.pi * x / length)

    velocity_expr = input("Enter initial velocity g(x), or Enter for zero: ").strip()
    if velocity_expr:
        def velocity(x):
            allowed = {"x": x, "math": math, "sin": math.sin, "cos": math.cos,
                       "exp": math.exp, "pi": math.pi}
            return float(eval(velocity_expr, {"__builtins__": {}}, allowed))
    else:
        velocity = lambda x: 0.0

    left = float(input("Boundary value at x=0: "))
    right = float(input("Boundary value at x=L: "))
    u = [[0.0] * (nx + 1) for _ in range(nt + 1)]
    for i in range(nx + 1):
        u[0][i] = displacement(i * h)
    for j in range(nt + 1):
        u[j][0] = left
        u[j][nx] = right

    for i in range(1, nx):
        xi = i * h
        u[1][i] = ((1 - lam2) * u[0][i]
                   + (lam2 / 2) * (u[0][i - 1] + u[0][i + 1])
                   + dt * velocity(xi))

    for j in range(1, nt):
        for i in range(1, nx):
            u[j + 1][i] = (
                2 * (1 - lam2) * u[j][i]
                + lam2 * (u[j][i - 1] + u[j][i + 1])
                - u[j - 1][i]
            )

    print("\nWave displacement table u(x,t):")
    print("Time\t" + "\t".join(f"x={i*h:.3f}" for i in range(nx + 1)))
    for j in range(nt + 1):
        print(f"{j*dt:.4f}\t" + "\t".join(f"{value:.6f}" for value in u[j]))


# ---------------------------------------------------------------------------
# 20B. HYPERBOLIC PDE: WAVE EQUATION WITH SINE INITIAL DISPLACEMENT
# ---------------------------------------------------------------------------

def wave_equation_sine():
    import numpy as np

    length = float(input("Enter string length L: "))
    speed = float(input("Enter wave speed c: "))
    M = int(input("Enter spatial intervals M: "))
    total_time = float(input("Enter final time T: "))
    N = int(input("Enter time intervals N: "))
    if length <= 0 or speed <= 0 or M < 2 or total_time <= 0 or N < 1:
        raise ValueError("Invalid input.")

    h = length / M
    dt = total_time / N
    r = speed * dt / h
    print(f"h = {h}, dt = {dt}, Courant number r = {r}")
    if r > 1:
        raise ValueError("Stability condition violated: r must be <= 1.")

    x = np.linspace(0, length, M + 1)
    def f(xv):
        return 0.01 * math.sin(math.pi * xv / length)
    def g(xv):
        return 0.0

    U = np.zeros((N + 1, M + 1))
    for i in range(M + 1):
        U[0, i] = f(x[i])
    for i in range(1, M):
        U[1, i] = (
            f(x[i]) + dt * g(x[i])
            + (r**2 / 2) * (f(x[i + 1]) - 2 * f(x[i]) + f(x[i - 1]))
        )

    U[:, 0] = U[:, M] = 0.0
    for n in range(1, N):
        for i in range(1, M):
            U[n + 1, i] = (
                2 * (1 - r**2) * U[n, i]
                + r**2 * (U[n, i - 1] + U[n, i + 1])
                - U[n - 1, i]
            )
        U[n + 1, 0] = U[n + 1, M] = 0.0

    mid = M // 2
    print(f"\nInitial midpoint displacement: {U[0, mid]:.8f}")
    print(f"First time-level midpoint displacement: {U[1, mid]:.8f}")
    print(f"\nFinal displacement at time T = {total_time}")
    print("x\t\tU(x,T)")
    step = max(1, M // 10)
    for i in range(0, M + 1, step):
        print(f"{x[i]:.4f}\t\t{U[N, i]:.8f}")
    if M % step != 0:
        print(f"{x[M]:.4f}\t\t{U[N, M]:.8f}")


# ---------------------------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------------------------

METHODS = [
    ("01. Gauss Elimination", gauss_elimination_basic),
    ("02. Gauss Elimination with Row Pivoting", gauss_elimination_pivoting),
    ("03. Gauss-Jacobi Method", gauss_jacobi),
    ("04. Gauss-Seidel Method", gauss_seidel),
    ("05. Power Method", power_method),
    ("06. Gauss-Jordan Matrix Inverse", jordan_inverse),
    ("07. Lagrange Interpolation", lagrange_interpolation),
    ("08. Newton Divided Difference", newton_divided_difference),
    ("09. Newton Forward Difference", newton_forward_difference),
    ("10. Newton Backward Difference", newton_backward_difference),
    ("11. Natural Cubic Spline", cubic_spline),
    ("12. Composite Trapezoidal Rule", trapezoidal_rule),
    ("13. Composite Simpson's 1/3 Rule", simpsons_one_third_rule),
    ("14. Two-Point Gauss Quadrature", gauss_quadrature_two_point),
    ("15. Adams-Bashforth-Moulton Predictor-Corrector", adams_bashforth_moulton),
    ("16. Milne Predictor-Corrector", milne_predictor_corrector),
    ("17. Runge-Kutta Fourth Order", runge_kutta_fourth_order),
    ("18A. Elliptic PDE - Liebmann Method", elliptic_pde_liebmann),
    ("18B. Elliptic PDE - Rectangular Mesh", elliptic_pde_rectangular_mesh),
    ("19A. Heat Equation - Custom FTCS/Bender-Schmidt", heat_equation_ftcs_custom),
    ("19B. Heat Equation - FTCS Sine Initial Condition", heat_equation_ftcs_sine),
    ("19C. Heat Equation - Crank-Nicolson", crank_nicolson_heat_equation),
    ("20A. Wave Equation - General Initial Conditions", wave_equation_finite_difference),
    ("20B. Wave Equation - Sine Initial Condition", wave_equation_sine),
]


def main():
    while True:
        print("\n" + "=" * 72)
        print("NUMERICAL METHODS - PYTHON REFERENCE")
        print("=" * 72)
        for index, (title, _) in enumerate(METHODS, 1):
            print(f"{index:2}. {title}")
        print(" 0. Exit")

        choice = input("\nSelect a method: ").strip()
        if choice == "0":
            print("Exiting.")
            break
        if not choice.isdigit() or not 1 <= int(choice) <= len(METHODS):
            print("Invalid selection. Try again.")
            continue

        title, method = METHODS[int(choice) - 1]
        print(f"\n--- {title} ---")
        try:
            method()
        except (ValueError, IndexError, ZeroDivisionError) as exc:
            print(f"\nCould not complete this method: {exc}")
        except KeyboardInterrupt:
            print("\nMethod cancelled by user.")
        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()
