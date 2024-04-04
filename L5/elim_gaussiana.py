#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
# M(fi:ff, ci:cf)

import numpy as np
import matplotlib.pyplot as plt

"""
i, j = 1, 1
while j< n:
    k = i
    while k < n:
        a_kj= (-1)*Ac[k][j-1]/Ac[i-1][i-1]
        cant_op+=2
        while j < n:
            Ac[k][j] += a_kj*Ac[i-1][j]
            j+=1
            cant_op+=3
        j = i
        Ac[k][j-1] = a_kj
        k+=1
    j+=1
    i+=1
    
def g(M):
    n = M.shape[0]
    #i = 0
    vec_col = np.ones(n)
    for j in range(n):
        for i in range(j,n):
            if i > j:
                M[j][i]= M[i][j]/M[j][j]
                vec_col[i] = M[i][j]/M[j][j]
            else:
                vec_col[i] = M[i][j]
            
        print(vec_col)
"""

def triang_submatriz(M):
    op=0
    n = M.shape[0]
    for i in range(1,n):
        temp = M[i][0]/M[0][0]
        op+=1
        for j in range (1,n):            
            M[i][j]+= (-1)*temp*M[0][j]
            op+=2
        M[i][0]= temp
    return op

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    for j in range(n-1):
        cant_op += triang_submatriz(Ac[j:,j:])
            
    L = np.tril(Ac,-1) + np.eye(A.shape[0]) 
    U = np.triu(Ac)

    return L, U, cant_op

def matriz_B(n, verbose=True):
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    if verbose:
        print('Matriz B \n', B)
    return B

def plots_operaciones_x_tamaño(n, verbose=True):
    y = np.zeros(n)
    
    for i in range(1,n+1):
        B = matriz_B(i, verbose=False)
        temp = elim_gaussiana(B)
        y[i-1] = temp[2]
    
    x = np.arange(n)
    if verbose:
        print(y)
        print(x)

    plt.title("Cantidad de operaciones por tamaño de matriz ")
    plt.xlabel("Iteraciones")
    plt.ylabel("Cantidad de operaciones")
    plt.plot(x, y)
    plt.show()
    plt.close()
    
def es_U(M):
    res = True
    n = M.shape[0]
    i = 1
    while i < n and res:
        res = 0 == np.sum(M[i][:i])
        i+=1
    return res
    
A = np.array([[2,1,2,3],[4,3,3,4],[-2,2,-4,-12],[4,1,8,-3]])
b = np.ones((4,1))        
#A = np.concatenate((A,b), axis=1)

def sustitucion_iter(A,x):
    j = A.shape[0]-1
    op = 0
    while j >= 0:
        while x[j][0] == 0 and j >=0:
            j-=1
            op+=1
        if j<0 : break
        if A[j][j] == 0 and x[j][0] != 0:
            print("sistema incompatible")
            break
        x[j][0] = x[j][0]/A[j][j]
        op+=1
        i = j-1
        while i >=0 :
            x[i][0]-=x[j][0]*A[i][j]
            op+=2
            i-=1
        j-=1
    
    return op
    
    
def sustitucion(A,b):
    n=A.shape[0]
    m=A.shape[1]
    x = b.shape[0]
    print("Condicion: ", n!=m or not es_U(A) or n!=x)
    if n!=m or not es_U(A) or n!=x:
        print('Matriz no cuadrada o no triangular superior o el vector b no es valido')
        return 0
    x = b.copy()
    cant_op = sustitucion_iter(A, x)
    
    if np.allclose(np.linalg.norm(b - A@x, 1), 0): 
        print("oki")
        return x, cant_op
    else: 
        print("salio como el qlo")
        print(x)
    
A = elim_gaussiana(A)
A = A[1]

x = sustitucion(A, b)


def main():
    n = 7
    B =matriz_B(n)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )
    
    n = 100
    plots_operaciones_x_tamaño(n)

if __name__ == "__main__":
    main()
    
    