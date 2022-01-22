##################
## Importations ##
##################

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from math import sqrt
from PIL import Image


###############
## Fonctions ##
###############

def m(X):
    if len(X) != 0:
        return sum(X)/len(X)

def power(n):
    def pown(x):
        return x**n
    return pown

def precision(f,X,Y): #corrélation
    fX = v(f)(X)
    EX = m(fX)
    EY = m(Y)
    XE = fX-EX
    YE = Y-EY
    cov = m(XE*YE)
    SX = sqrt(m(XE**2))
    SY = sqrt(m(YE**2))
    return (cov/(SX*SY))**2


##################
## Modélisation ##
##################

def v(f): #Polynome
    n = len(f)
    def vf(x):
        return sum([f[i]*(x**i) for i in range(n)])
    return vf

def model(n,X,Y): #Regression polynomiale recursive
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    if n==1:
        pente = (m(X*Y)-m(X)*m(Y))/(m(X*X)-m(X)*m(X))
        origine = m(Y) - pente*m(X)
        f = [origine,pente]
        return f
    L = len(X)
    Xbis = np.array([(X[i+1]+X[i])/2 for i in range(L-1)])
    Ybis = np.array([(Y[i+1]-Y[i])/(X[i+1]-X[i]) for i in range(L-1)])
    fo = model(n-1,Xbis,Ybis)
    f = [0] + [fo[i]/(i+1) for i in range(n)]
    #print("f;", f, "f[0]:", Y-v(f)(X))
    #print("Y:", Y, "X:", X)
    f[0] = m(Y-v(f)(X))
    return f

def all(n,X,Y): #Affichage de l'approche polynomiale
    p = 200
    Xl = np.linspace(X[0],X[-1],int(p*(X[-1]-X[0])))
    f = model(n,X,Y)
    p = precision(f,X,Y)
    print("f:", f, "precision: "+str(p*100)+"%")
    plt.scatter(X,Y)
    plt.plot(Xl,v(f)(Xl))
    plt.show()
    return p, f




X = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64) #N < len(X)
Y = np.array(v([0, 0, 1])(X), dtype=np.float64) #x**2
Y += np.random.randn(len(X)) #Ajout de bruit
for i in range(1,6):
    print("N :",i)
    all(i,X,Y)

