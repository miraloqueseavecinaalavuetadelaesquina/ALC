#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP 1
Grupo ALV
"""
import sys
import numpy as np
import matplotlib as ptl
from progress.bar import Bar
# Cambiar al root_dir local
root_dir = '/home/antonyus/ALC/TP1'
# Agregamos el root_dir del directorio donde está ubicado nuestro módulo a importar
sys.path.append(root_dir)
import func as fn

def main():
    
    # Ejercicio 4
    A = 2 * np.random.rand(10,10) -1
    b = 2 * np.random.rand(10,1) - 1
    x = fn.resolverLU(A, b)
    Ax_b = A@x - b
    norma_Ax_b = fn.norma2Vectorial(Ax_b)
    norma_x = fn.norma2Vectorial(x)
    e = norma_Ax_b/norma_x
    
    # Ejercicio 5
    A = 2 * np.random.rand(10,10) -1
    A_i = fn.inversa(A)
    Ac = A@A_i - np.eye(A.shape[0])
    norm_frobenius = fn.normaF2(A)
    
    # Ejercicio 7
    print("Ej 7 \n")
    #Definimos diccionario de funiones
    funciones = {'inv' : fn.inversa,
                 'solve' : fn.resolverLU,
                 'norma' : fn.norma2Vectorial}
    v1,v2 = fn.perfomance(funciones,rango_de_iteracion=(10,200))
    dv1 = {'vector': v1,
           'leyenda': "Resolver (LU)x=b"}
    dv2 = {'vector': v2,
           'leyenda': "x = inversa(A)@b"}
    rotulos = {'titulo' : 'Errores relativos $\|Ax-b\|_2/\|b\|_2$',
               'eje x' : 'Dimensión de matriz',
               'eje y' : '$\sum_{i = 0}^10{ln(\|A_i*x_i-b_i\|_2/\|b_i\|_2)}$'}
    
    fn.plotearVectores(dv1, dv2, rotulos)
    
    rotulos['titulo'] = 'Distancia errores relativos $\|Ax-b\|_2/\|b\|_2$'
    
    fn.plotDistanciaVectores(v1, v2, rotulos)
    
    
    # Ejercicio 8
    print("Ej 8 \n")
    v1,v2 = fn.perfomance(funciones,rango_de_iteracion=(10,200), er_b=False)
    dv1 = {'vector': v1,
           'leyenda': "Resolver (LU)x=b"}
    dv2 = {'vector': v2,
           'leyenda': "x = inversa(A)@b"}
    rotulos = {'titulo' : 'Errores relativos $\|xp-x\|_2/\|x\|_2$',
               'eje x' : 'Dimensión de matriz',
               'eje y' : '$\sum_{i = 0}^10{ln(\|xp_i-x_i\|_2/\|x_i\|_2)}$'}
    
    fn.plotearVectores(dv1, dv2, rotulos)
    
    rotulos['titulo'] = 'Distancia errores relativos $\|xp_i-x_i\|_2/\|x_i\|_2$'
    
    fn.plotDistanciaVectores(v1, v2, rotulos)
    
    # Ejercicio 9
    print("Ej 9 \n")
    funciones = {'inv' : np.linalg.inv,
                 'solve' : np.linalg.solve,
                 'norma' : fn.norma2Vectorial}
    v1,v2 = fn.perfomance(funciones,rango_de_iteracion=(10,200))
    dv1 = {'vector': v1,
           'leyenda': "np.linalg.solve Ax=b"}
    dv2 = {'vector': v2,
           'leyenda': "x = np.linalg.inv(A)@b"}
    rotulos = {'titulo' : 'Errores relativos $\|Ax-b\|_2/\|b\|_2$',
               'eje x' : 'Dimensión de matriz',
               'eje y' : '$\sum_{i = 0}^10{ln(\|A_i*x_i-b_i\|_2/\|b_i\|_2)}$'}
    
    fn.plotearVectores(dv1, dv2, rotulos)
    
    rotulos['titulo'] = 'Distancia errores relativos $\|Ax-b\|_2/\|b\|_2$'
   
    fn.plotDistanciaVectores(v1, v2, rotulos)
    
    # Ejercicio 10
    print("Ej 10 \n")
    funciones = {'inv' : fn.inversa,
                 'solve' : fn.resolverLU,
                 'norma' : fn.norma2Vectorial}
    v1,v2 = fn.perfomance(funciones,rango_de_iteracion=(2,20),tipo_de_matriz='H')
    dv1 = {'vector': v1,
           'leyenda': "Resolver (LU)x=b"}
    dv2 = {'vector': v2,
           'leyenda': "x = inversa(A)@b"}
    rotulos = {'titulo' : 'Errores relativos $\|Ax-b\|_2/\|b\|_2$',
               'eje x' : 'Dimensión de matriz',
               'eje y' : '$\sum_{i = 0}^10{ln(\|A_i*x_i-b_i\|_2/\|b_i\|_2)}$'}
    
    fn.plotearVectores(dv1, dv2, rotulos)
    
    rotulos['titulo'] = 'Distancia errores relativos $\|Ax-b\|_2/\|b\|_2$'
   
    fn.plotDistanciaVectores(v1, v2, rotulos)
    
    
    # Ejercicio 11
    #Definimos diccionario de funiones
    print("Ej 11\n")
    funciones = {'inv' : fn.inversa,
                 'solve' : fn.resolverLU,
                 'norma' : fn.norma2Vectorial}
    v1,v2 = fn.perfomance(funciones,rango_de_iteracion=(2,40), tipo_num=np.float16)
    dv1 = {'vector': v1,
           'leyenda': "Resolver (LU)x=b"}
    dv2 = {'vector': v2,
           'leyenda': "x = inversa(A)@b"}
    rotulos = {'titulo' : 'Errores relativos $\|Ax-b\|_2/\|b\|_2$',
               'eje x' : 'Dimensión de matriz',
               'eje y' : '$\sum_{i = 0}^10{ln(\|A_i*x_i-b_i\|_2/\|b_i\|_2)}$'}
    
    fn.plotearVectores(dv1, dv2, rotulos)
    
    rotulos['titulo'] = 'Distancia errores relativos $\|Ax-b\|_2/\|b\|_2$'
   
    fn.plotDistanciaVectores(v1, v2, rotulos)
    
    return

if __name__ == "__main__":
    main()
    




