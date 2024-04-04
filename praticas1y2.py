#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts practica 1 y 2
Created on Sat Sep  9 20:32:08 2023
@author: antonyus
Materia:ALC
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
import rsrs as rs 

rootdir = '~/ALC'
# Agregamos el root_dir del directorio donde está ubicado nuestro módulo a importar
sys.path.append(rootdir)

#****************************** Practica 1 ************************************

def traza(M):
    # condicion para asegurar entrada de tipo np.ndarray
    
    if M.shape[0] == M.shape[1] or M.shape[0] == 1:
        traza = 0
        for i in range(M.shape[0]):
            traza += M[i][i]
        return traza
    else :
        print("Matriz no cuadrada")

    return 0

def sumatoria_M(M):
    s = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            s += M[i][j]
    return s

#************************** conceptos básicos *********************************

# Máximo número flotante que puede representar Python:
print('Máximo número flotante que puede representar Python: ', np.finfo(float).max, '\n')

# Mínimo flotante positivo normalizado que puede representar Python:
print('Mínimo flotante positivo [normalizado] que puede representar Python: ', np.finfo(float).tiny, '\n')

# Mínimo flotante positivo [subnormal] que puede representar Python:
print('Mínimo flotante positivo [subnormal] que puede representar Python: ', np.nextafter(0., 1.), '\n')

# Epsilon de máquina
print("Epsilon de máquina float: ", np.finfo(float).eps)
print("Epsilon de máquina double: ", np.finfo(np.double).eps)

#****************************** Practica 2 ************************************

# Ej 6

# a
p = 1e34
q = 1
print(p+q-p) # resultado esperado = 1e34
print(f"1e34 + 1 = {1e34:.17f} + {1:.17f} = {1e34+1:.17f}")
print(f"{np.single(p):.17f}")


#b
p = 100
q = 1e-15
print((p+q)+q)
print(f"(100 + 1e-15) + 1e-15 = ({100:.17f} + {1e-15:.17f}) + {1e-15:.17f} = {(100+1e-15)+1e-15:.17f}")
print(((p+q)+q)+q) # elimina por redondeo

#c
print(0.1+0.2 == 0.3)
print(f"0.1 + 0.2 = {0.1:.17f} + {0.2:.17f} = {0.1+0.2:.17f}")
print(0.1+0.2)

#d
print(0.1+0.3 == 0.4)
print(f"0.1 + 0.3 = {0.1:.17f} + {0.3:.17f} = {0.1+0.3:.17f}")

#e
print(1e-323)
print(f"{1e-323:.327f}")
#f
print(1e-324)
print(f"{1e-324:.324f}")

eps = np.finfo(float).eps
print(f"{np.single(eps):.17f}")

# g
print(eps/2)

#h
print((1+eps/2)+eps/2)

#i
print(1+(eps/2+eps/2))

#j
print(((1+eps/2)+eps/2)-1)

#k
print((1+(eps/2+eps/2))-1)

#%%
#l
print("sen(pi*10^j) \n")
n = 25
xy = np.zeros((n,n))
for j in range(n):
    t = np.pi*10**j
    ft = np.sin(t)
    xy[0][j]=t
    xy[1][j]=ft
    print("j = "+str(j+1)+" -> ", np.sin(np.pi*10**j+1))

x = np.arange(0,n)

#plt.yscale("log")
plt.title("sen(pi*10^j)")
plt.xlabel("Iteraciones")
plt.ylabel("Variación f(t)")
plt.plot(x, xy[1])
plt.show()
plt.close()

plt.title("pi*10^j")
plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Variación t")
plt.plot(x, xy[0])
plt.show()
plt.close()

#%%


#m
print("sen(pi/2 + pi*10^j) \n")
for j in range(n):
    t = np.pi/2 + np.pi*10**j+1
    ft = np.sin(t)
    xy[0][j]=t
    xy[1][j]=ft
    print("j = "+str(j+1)+" -> ", np.sin(np.pi/2 + np.pi*10**j+1))

plt.title("sen(pi/2 + pi*10^j)")
plt.xlabel("Iteraciones")
plt.ylabel("Variación f(t)")
plt.plot(x, xy[1])
plt.show()
plt.close()

plt.title("pi/2 + pi*10^j")
plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Variación t")
plt.plot(x, xy[0])
plt.show()
plt.close()

#%%
# Ejerecicio 7

n = 7
s = np.float32(0)
xy = np.zeros(n)
for i in range(1,10**n+1):    
    s += 1/i
    xy[i] = s
    

#%%
# Ejercicio 8
eps = 0.0001
m_A = np.array([[1,2,1],[2,3-eps,2+eps],[0,1+eps,eps]])
b = np.array([[0],[0.1],[0.1]])
x = np.linalg.solve(m_A, b)



#%%
# ejercicio 9 

m = 1e4
m_A = np.array([[1,m,5*m],[1,3*m,3*m],[1,m,2*m]])
b = np.array([[2*m/3],[2*m/3],[m/3]])
x = np.linalg.solve(m_A, b)
print(type(x))
x.size
# Experimentamos para diversos m
n = 50
m = 0
xy = np.zeros((n,n,n))
while m < n :
    x = np.linalg.solve(m_A, b)
    for i in range(x.size):
        xy[i][m]=x[i]
    m+=1

print(m_A@x)

x = np.arange(m)

plt.title("Ax=b condicionada a n ")
plt.xlabel("Iteraciones")
plt.ylabel("Variación Ax=b")
plt.plot(x, xy[0])
plt.plot(x, xy[1])
plt.show()
plt.close()


#%%
# Ejercicio 17

"""
A : R^(nxn) 
estimar la norma 2 para matrices condicionadas por vectores ||A||_2, entre varios v!=0 generados al azar

generar los 100 primeros terminos de la suseción :
     s_1 = 0
     s_{k+1} = max{s_k, ||Ax_k||_2/||x_k||_2}, donde x_k : R³ y x!=0 

"""

def aproxNorma2xsuseciones(M, it = 100 ):
    
    if M.shape[0] == M.shape[1]:
        
        V = np.random.rand(M.shape, it)
        # Generamos sucesion
        s = 0
    
    return 0

p = np.random.rand(3,4) - 0.5

#%%

a = np.array([[1,1,2]])

print(a.shape)