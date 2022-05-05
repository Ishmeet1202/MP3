from scipy.special import roots_laguerre
import math
import pandas as pd


def MyLaguQuad(f, n):
    roots, weights = roots_laguerre(n)
    W = list(weights)
    R = list(roots)
    I = []

    for i, j in zip(W, R):
        I.append(i * f(j) * p(j))
    return sum(I)


n = 2
N = []
for i in range(1, 8):
    N.append(2 ** i)
val1 = []
val2 = []
for i in range(2):
    print("Function",i+1)
    func = eval("lambda x:" + input("Function :"))
    power = eval("lambda x:" + input("Exponent of e :"))
    for n in N:
        roots, weights = roots_laguerre(n)

        f = lambda x: func(x)
        p = lambda x: math.exp(power(x))

        W = list(weights)
        R = list(roots)

        if i == 0:
            val1.append(MyLaguQuad(f, n))
        else:
            val2.append(MyLaguQuad(f, n))

dtf1 = pd.DataFrame({"n": N, "I1": val1, "I2": val2})
print(dtf1)
