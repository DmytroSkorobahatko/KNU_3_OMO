def newton(f, Df, x0, epsilon, N):
    """
    Newton's method.

    f : function
    Df : function - Derivative of f(x).
    x0 : number - Guess for a solution.
    epsilon : number - Stopping criteria.
    N : integer - Maximum iterations.

    xn, n : solution, iterations
    """
    xn = x0
    n = 1
    for n in range(N):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print(f"x = {xn} \nafter {n} iterations")
            return None
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn / Dfxn
    print(f"x = {xn} \nafter {n} iterations")  # max iterations


def bisection(f, a, b, N):
    """
    Bisection method.

    f : function - to approximate a solution f(x)=0.
    a,b : numbers - the interval in which to search for a solution.
    N : integer - number of iterations.

    x, i : solution, iterations
    """
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    x, i = a, 1
    for n in range(N + 1):
        i = n
        m_n = (a_n + b_n) / 2
        f_m_n = f(m_n)
        if f_m_n == 0:
            x = m_n
            print(f"x = {x} \nafter {i} iterations")
        elif f(a_n) * f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n) * f_m_n < 0:
            a_n = m_n
            b_n = b_n
        else:
            print("Bisection failed to approx")
            break
    x = (a_n + b_n) / 2
    print(f"x = {x} \nafter {i} iterations")


def relaxation(f, x, epsilon, N):
    """
    Relaxation method.

    f : function
    x : number - Guess for a solution f(x).
    epsilon : number - Stopping criteria abs(f(x)) < epsilon.
    N : integer - Maximum iterations.
    values : list - x's values.

    x, i : solution, iterations
    """
    values = []
    for i in range(N):
        values.append(x)
        x = f(x)
        if abs(x - values[-1]) < epsilon:
            print(f"x = {x} \nafter {i} iterations")
            break
    else:
        print("Relaxation failed to approx")


def run(method_name):
    """
    func = x^3 - x^2 - 1
    sol = 1.4655 7123 1876 7682
    """
    print('\n', method_name)
    match method_name:
        case 'Newton':  # f, Df, x0, epsilon, N | f(x) = 0
            f = lambda x: x ** 3 - x ** 2 - 1  # p(x) = x^3 - x^2 - 1
            Df = lambda x: 3 * x ** 2 - 2 * x
            newton(f, Df, 0.1, 1e-13, 100)

        case 'Bisection':  # f, a, b, N | f(x) = 0
            f = lambda x: x ** 3 - x ** 2 - 1  # f(x) = x^3 - x^2 - 1
            bisection(f, 1, 2, 45)

        case 'Relaxation':  # f, x, epsilon, N | x = g(x) = x + T*f(x)
            f = lambda x: x - 0.1 * (x ** 3 - x ** 2 - 1)  # x = x - 0.1 * (x^3 - x^2 - 1)
            relaxation(f, 1, 1e-14, 100)


if __name__ == '__main__':
    '''there are "Newton", "Bisection" and "Relaxation" methods'''
    run('Newton')
    run('Bisection')
    run('Relaxation')
