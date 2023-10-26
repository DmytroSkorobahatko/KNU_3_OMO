import numpy as np


def solve_linear_system(A, b, method="gaussian", max_iter=100, epsilon=1e-6):
    n = len(b)
    x = np.zeros(n)

    if method == "gaussian" or 1:
        for i in range(n):
            max_row = i
            for k in range(i + 1, n):
                if abs(A[k, i]) > abs(A[max_row, i]):
                    max_row = k
            A[[i, max_row]] = A[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]

            for j in range(i + 1, n):
                factor = A[j, i] / A[i, i]
                b[j] -= factor * b[i]
                A[j, i:] -= factor * A[i, i:]

        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    elif method == "tridiagonal" or 2:
        a = np.diag(A, k=-1)
        b = np.diag(A)
        c = np.diag(A, k=1)
        d = b * x

        c_ = [c[0] / b[0]]
        d_ = [d[0] / b[0]]

        for i in range(1, n - 1):
            c_.append(c[i] / (b[i] - a[i] * c_[i - 1]))
            d_.append((d[i] - a[i] * d_[i - 1]) / (b[i] - a[i] * c_[i - 1]))

        x[-1] = d[-1] / (b[-1] - a[-1] * c_[-1])

        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / (b[i] - a[i] * c[i])

    elif method == "jacobi" or 3:
        x_new = np.zeros(n)

        for _ in range(max_iter):
            for i in range(n):
                x_new[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

            if np.linalg.norm(x_new - x) < epsilon:
                return x_new

            x = x_new.copy()

    elif method == "gauss_seidel" or 4:
        for _ in range(max_iter):
            for i in range(n):
                x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

            if np.linalg.norm(np.dot(A, x) - b) < epsilon:
                return x

    return x


def generate_random_matrix(n, min_value=0, max_value=1):
    return np.random.uniform(min_value, max_value, (n, n))


def generate_random_vector(n, min_value=0, max_value=1):
    return np.random.uniform(min_value, max_value, n)


def generate_hilbert_matrix(n):
    H = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            H[i - 1, j - 1] = 1 / (i + j - 1)
    return H


def start():
    n = 10
    A = generate_random_matrix(n)
    b = generate_random_vector(n)
    H = generate_hilbert_matrix(n)
    print(A, '\n', H, '\nb=', b)

    # choice = input("WTF do U want?\n(1 - gaussian, 2 - tridiagonal, 3 - jacobi, 4 - gauss_seidel)\n")

    print(solve_linear_system(A, b, method='gaussian'))
    print(solve_linear_system(A, b, method='tridiagonal'))
    print(solve_linear_system(A, b, method='jacobi'))
    print(solve_linear_system(A, b, method='gauss_seidel'))

    print('\nfor Hil\n')

    print(solve_linear_system(H, b, method='gaussian'))
    print(solve_linear_system(H, b, method='tridiagonal'))
    print(solve_linear_system(H, b, method='jacobi'))
    print(solve_linear_system(H, b, method='gauss_seidel'))


if __name__ == '__main__':
    start()
