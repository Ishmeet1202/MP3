from pip import main
from sympy import *


def f1(x,y):
    return lambda x,y : x + y

t=f1(3,2)
print(t(3,2))

def f1(f):
    x = var('x')
    y = var("y")
    expr = sympify(f)
    integrand = lambdify((x,y),expr)
    value = integrand(2,3)
    return value

z = f1("x")
print(z)



# if __name__ == "__main__":
#     print(f1(3))
