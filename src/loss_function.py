import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-ticks")
from sklearn.decomposition import PCA
from numpy.random import randn # Gaussian random numbe
#Define the constants and functions here
m = 20
sigma = 2
sigma2 = 2*sigma**2
w = randn(m) / sigma
b = np.random.rand(m) *2*np.pi

n = 200
x = randn(n)/2
#y = 1 + 5*np.sin(x/10) + 5*x**2 + randn(n)
y=np.sin(x) + randn(n)
lam = 0.000001

'''
Different kernels
'''
# Rahimi and Recht 
def z_RR(x):
    return np.sqrt(2)*np.cos(w*x + b)

def K_RR(x, y):
    return 1/m * np.sum(z_RR(x)*z_RR(y))

# Suzuki
def zc(x):
    return 1/np.sqrt(m) * np.cos(w*x)

def zs(x):
    return 1/np.sqrt(m) * np.sin(w*x)

def K_S(x, y):
    return np.sum(zc(x) * zc(y)) + np.sum(zs(x) * zs(y))

# Gaussian kernel
def K_G(x, y): 
    return np.exp((-(x-y)**2)/sigma2)

# Find the constant a 
def K(k, x, y):
    n = len(x)
    Kxy = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            Kxy[i, j] = k(x[i], y[j])
    return Kxy

# Find the constant a 
def a_G(x, y):
    n = len(x)
    Kxx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Kxx[i, j] = K_G(x[i], x[j])
    return np.linalg.inv(Kxx + lam*np.eye(n)) @ y

def a_S(x, y):
    n = len(x)
    Z = np.zeros((n,m))
    for i in range(n):
        Z[i,:] = zc(x[i]) + zs(x[i])
    beta = np.linalg.inv((Z.T@Z) + lam*np.eye(m)) @ Z.T @ y
    return beta

def a_RR(x, y):
    n = len(x)
    Z = np.zeros((n,m))
    for i in range(n):
        Z[i,:] = z_RR(x[i])
    beta=np.dot(np.linalg.inv(np.dot(Z.T, Z)+lam*np.eye(m)), np.dot(Z.T,y))
    return beta

# Find norm f^2
def norm_f2(K, a):
    return a.T @ K @ a

# With Riemann and Recht 
alpha_RR = a_RR(x, y)
Ky_RR = K(K_RR, x, y)
Kx_RR = K(K_RR, x, x)
#normf_RR = norm_f2(Ky_RR, alpha_RR)

# With Suzuki
alpha_S = a_S(x, y)
Ky_S = K(K_S, x, y)
Kx_S = K(K_S, x, x)
#normf_S = norm_f2(Ky_S, alpha_S)

# With Gaussian kernel
alpha_G = a_G(x, y)
Ky_G = K(K_G, x, y)
Kx_G = K(K_G, x, x)
#normf_G = norm_f2(Ky_G, a_G)

def loss_function(k, a, x, y, normf):
    n = len(x)
    sum = 0
    for i in range(n):
        f_i = 0
        for j in range(n):
            f_i += a[j]*k(x[i], x[j])
        sum += (f_i - y[i])**2
    return 1/len(x) * sum + lam *normf

#loss_S = loss_function(K_S, a_S, x, y, normf_S)

#loss_G = loss_function(K_G, a_G, x, y, normf_G)

#loss_RR = loss_function(K_RR, a_RR, x, y, normf_RR)
# ikke samme alpha på cos og sin i rahimi
#print("Loss Suzuki: ", loss_S)
#print("Loss Gaussaian: ", loss_G)

r = np.sort(x)
u = np.zeros(n); v = np.zeros(n); s = np.zeros(n)
for j in range(n):
    S = 0
    for i in range(n):
        S = S + alpha_G[i] * K_G(x[i], r[j])
        u[j] = S; 
        v[j] = np.sum(alpha_S * (zc(r[j])+zs(r[j])))
        s[j] = np.sum(alpha_RR * z_RR(r[j]))
        
plt.scatter(x,y, facecolors ='none' , edgecolors = "k" , marker = "o")
plt.plot(r, u, c="r", label="with Gaussian")
plt.plot(r,v, c="b", label="with Suzuki")
plt.plot(r, s, c="y", label="with Rahimi")
plt.xlim ( -1.5 , 2)
plt.ylim ( -2 , 8)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Kernel Regression")
plt.legend( loc = "upper left" , frameon = True , prop ={'size': 14 } )
plt.show()