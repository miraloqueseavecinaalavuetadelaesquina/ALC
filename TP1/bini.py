#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 09:16:30 2023

@author: antonyus
"""
"""
def perfomance(dict_func, rango_de_iteracion=(2,10), ciclos=10, tipo_de_matriz='random', tipo_num=np.float64 ,er_b=True, verbose=True):
    v1 = np.zeros(rango_de_iteracion[1]+1, dtype=tipo_num)
    v2 = np.zeros(rango_de_iteracion[1]+1, dtype=tipo_num)
    if tipo_de_matriz == 'H':
        v3 = np.zeros(rango_de_iteracion[1]+1)
    elif tipo_de_matriz == 'random':
        v3 = np.zeros((3,rango_de_iteracion[1]+1))
        
    if verbose: bar = Bar('Procesando', max=rango_de_iteracion[1]+1, suffix='%(percent)d%%')    
    for i in range(rango_de_iteracion[0],rango_de_iteracion[1]+1):
        e1,e2, e3 = tipo_num(0), tipo_num(0), tipo_num(0)
        if tipo_de_matriz == 'random':
            minK, maxK = tipo_num(0), tipo_num(0)
        for j in range(ciclos):
            A = generarMatriz(fil=i,col=i,metodo=tipo_de_matriz, p=tipo_num)
            x = (2 * np.random.rand(i,1) - 1).astype(tipo_num)
            b = A@x
            A_i = dict_func['inv'](A)
            xp = dict_func['solve'](A,b)
            e = errorRelativo(M=A, v_x=x, v_xp=xp, v_b=b, funcion_norma=dict_func['norma'])
            e1 += e #
            xp = A_i@b
            e = errorRelativo(M=A, v_x=x, v_xp=xp, v_b=b, funcion_norma=dict_func['norma'], condicion=False)
            e2 += e
            if A.dtype == np.float64: e = np.linalg.cond(A)
            if tipo_de_matriz=='random': acotarK(inf=minK, sup=maxK, k=e)
            e3+=e
            
        v1[i] = e1
        v2[i] = e2
        v3
        if verbose: bar.next()
    if verbose:
        bar.finish()
        print()
        print("Finalizó con exito")
              
    return v1, v2

"""