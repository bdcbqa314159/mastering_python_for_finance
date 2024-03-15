#pylint: disable-all

import numpy as np

def incremental_search(f,a,b,dx):
    fa = f(a)
    c = a+dx
    fc = f(c)
    n = 1

    while np.sign(fa) == np.sign(fc):
        if a>=b:
            return a-dx,n
        a = c
        fa = f(a)
        c = a+dx
        fc = f(c)
        n+=1

    if fa == 0:
        return a, n
    elif fc == 0:
        return c,n
    else:
        return a+(c-a)/2, n
    

def bisection(f, a,b,tol = 0.01, max_iter = 10):
    
    c = 0
    n = 1
    while n<=max_iter:
        c = a + (b-a)/2
        if f(c) == 0. or abs(a-b)<tol:
            return c,n
        
        n+=1
        if f(c)<0:
            a = c
        else:
            b = c
    return c,n

def newton(f, df, x, tol = 0.01, max_iter = 100):
    n = 1
    while n<=max_iter:
        x1 = x-f(x)/df(x)

        if abs(x1-x)<tol:
            return x,n
        else:
            x = x1
            n+=1
    
    return None,n

def secant(f, a,b, tol = 0.01, max_iter = 100):
    n = 1
    while n<=max_iter:
        c = b - f(b)*((b-a)/(f(b)-f(a)))
        if abs(b-c)<tol:
            return c,n
        a = b
        b = c
        n+=1

    return None, n

if __name__ == "__main__":
    pass

