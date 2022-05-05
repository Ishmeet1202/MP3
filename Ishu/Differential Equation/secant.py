import numpy as np

def funct(x):
    y= x**2 -x -1 
    return y

def SecantMethod(x0,x1,epsilon):
    n=0

    while abs((x0-x1)/x0) > epsilon:
        x2 = x1 - funct(x0)* (x1-x0)/(funct(x1)-funct(x0))
        x0 = x1
        x1 = x2
        
        n += 1
    return x0,n
    
if __name__ == "__main__":
    sol = SecantMethod(-10, 10, 0.0001)
    print(sol)