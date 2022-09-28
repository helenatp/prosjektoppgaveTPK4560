import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use("seaborn-ticks")
from sklearn.decomposition import PCA
from numpy.random import randn # Gaussian random numbers
from scipy.stats import norm

#Define the constants and functions here
m = 20
sigma = 2
sigma2 = 2*sigma**2
w = randn(m) / sigma
b = np.random.rand(m) *2*np.pi

n = 200
x = randn(n)/2
# y = 1 + 5*np.sin(x/10) + 5*x**2 + randn(n)
y=np.sin(x) + randn(n)
x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)
lam = 0.0001

'''
Different kernels
'''
# Rahimi and Recht 
def z_RR(x):
    return np.sqrt(2)*np.cos(w*x + b)

def K_RR(x, y):
    return 1/m * np.sum(z_RR(x) * z_RR(y))

# Suzuki
def zc(x):
    return 1/np.sqrt(m) * np.cos(w * x)

def zs(x):
    return 1/np.sqrt(m) * np.sin(w * x)

def K_S(x, y):
    return np.sum(zc(x) * zc(y)) + np.sum(zs(x) * zs(y))

# Exact Gaussian kernel
def K_G(x, y): 
    return np.exp(-(x-y)**2/sigma2)

#print(K_G(2, 3))

# Find the constant a 
def K(k, x, y):
    m = len(x)
    K = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            K[i, j] = k(x[i], y[j])
    return K

def a(k, x, y):
    m = len(x)
    K = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            K[i, j] = k(x[i], x[j])
    return np.linalg.inv(K + m*lam*np.eye(m)) @ y

# Find norm f^2
def norm_f2(K, a):
    return a.T @ K @ a

# With Riemann and Recht 
a_RR = a(K_RR, x, y)
Ky_RR = K(K_RR, x, y)
Kx_RR = K(K_RR, x, x)
normf_RR = norm_f2(Ky_RR, a_RR)

# With Suzuki
a_S = a(K_S, x, y)
Ky_S = K(K_S, x, y)
Kx_S = K(K_S, x, x)
normf_S = norm_f2(Ky_S, a_S)

# With exact kernel
a_G = a(K_G, x, y)
Ky_G = K(K_G, x, y)
Kx_G = K(K_G, x, x)
normf_G = norm_f2(Ky_G, a_G)

x_sort = np.sort(x)

def loss_function(Kx, a, x, y, normf):
    n = len(x)
    sum = 0
    for i in range(n):
        sum += (a[i]*Kx[i] - y[i])**2
    return 1/len(x) * sum + lam *normf

loss_S = loss_function(Kx_S, a_S, x, y, normf_S)

loss_G = loss_function(Kx_G, a_G, x, y, normf_G)

loss_RR = loss_function(Kx_RR, a_RR, x, y, normf_RR)
print("Loss Suzuki: ", loss_S)
print("Loss Gaussaian: ", loss_G)

plt.scatter(x,y, facecolors ='none' , edgecolors = "k" , marker = "o")
plt.plot( x_sort, loss_G, c = "r" , label = "w/o Approx")
plt.plot( x_sort, loss_S, c = "b" , label = "with Approx")
plt.plot(x_sort, loss_RR, c = "y", label ="With Rahimi")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Kernel Regression")
plt.legend( loc = "upper left" , frameon = True , prop ={'size': 14 } )
plt.show()