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
    max_value = n
    rand_matrix = np.random.uniform(min_value, max_value, (n, n))
    return rand_matrix


def generate_hilbert_matrix(n):
    H = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            H[i - 1, j - 1] = 1 / (i + j - 1)
    return H


def generate_vector(n, min_value=0, max_value=1):
    rand_vector = np.random.uniform(min_value, max_value, n)
    return rand_vector


def generate_answer_vector(n, A):
    ones = np.ones(n)
    vec = np.dot(A, ones)
    return vec


def use_all_methods(matrix, b):
    gaussian = solve_linear_system(matrix, b, method='gaussian')
    tridiagonal = solve_linear_system(matrix, b, method='tridiagonal')
    jacobi = solve_linear_system(matrix, b, method='jacobi')
    seidel = solve_linear_system(matrix, b, method='seidel')

    return [gaussian, tridiagonal, jacobi, seidel]


def start():
    n = 3
    A = generate_random_matrix(n)
    H = generate_hilbert_matrix(n)
    b = generate_vector(n)  # n, min_value=0, max_value=1
    print('A =', A, '\nH =', H)

    print('\nfor rand matrix')
    [print(i) for i in use_all_methods(A, b)]
    Ab = generate_answer_vector(n, A)
    [print(i) for i in use_all_methods(A, Ab)]

    print('\nfor Hil`s matrix')
    [print(i) for i in use_all_methods(H, b)]
    Hb = generate_answer_vector(n, H)
    [print(i) for i in use_all_methods(H, Hb)]


if __name__ == '__main__':
    start()
