# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 11:32:53 2024

@author: 373591
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import eigh
#%matplotlib inline
from scipy.special import binom
import time as tm
import itertools


def make_basis(N,k):
    V=[]
    for subset in itertools.combinations(np.arange(N+k-1),k-1):
        subset0=[]
        subset0.append(subset[0])
        for j in range(k-2):
            subset0.append(subset[j+1]-subset[j]-1)
        subset0.append(N+k-2-subset[-1])
        V.append(subset0)
    Basis=V[::-1]
    Ind={}
    d=len(V)
    for i in range(d):
        x=[int(t) for t in Basis[i]]
        name=str(x)
        Ind.update({name:i})
    return Basis, Ind


def multinomial(lst):
    res, i = 1, 1
    k=len(lst)
    l=0
    for a in lst:
        for j in range(1,a+1):
            #print(i/1)
            res *= i/k
            l+=1
            #res //= j
            res /=j
            
            i += 1
    #print(l, k)
    return res


def common_matrices(N, k):
    M={}
    Basis, Ind=make_basis(N, k)
    d=len(Basis)
    
    szd=[]
    for i in range(d):
        szd.append(np.sum(np.array(Basis[i])*(np.arange(k)-(k-1)/2)))
        
    
    Sz=np.diag(szd)
    Interaction=[]
    for l in range(k):
        n2=[]
        for i in range(d):
            n2.append(Basis[i][l]**2-Basis[i][l])
        Interaction.append(np.array(n2))
        
    Interaction=np.array(Interaction)
    
    

    
    Hopping=np.zeros((d,d), dtype=complex)
    for i in range(d):
        V=Basis[i].copy()
        
        for site in range(k-1):
            
            V2=V.copy()
            #V2=V
            #print(V)
            V2[site]=V2[site]-1
            V2[site+1]=V2[site+1]+1
            #print(V, V2, Basis[i])
            
            try:
                j=Ind[str([int(x) for x in V2])]
                Hopping[i, j]=np.sqrt(V[site]*V2[site+1])
                Hopping[j, i]=np.sqrt(V[site]*V2[site+1])
                #print(i,j)
            except:
                0
                

    
   
        
    
    

    M["Sz"]=Sz
    M["Hopping"]=Hopping
    M["Interaction"]=Interaction
    return M


def common_states(N, k):
    S={}
    Basis, Ind=make_basis(N, k)
    d=len(Basis)
    
    coherent=np.zeros(d, dtype=complex)
    for i in range(d):
        coeff=multinomial(Basis[i])
        coherent[i]=np.sqrt(coeff)#/np.sqrt(k**N)

    S["SQL"]=coherent
    return S




def fisher_info_pure(psi, H):
    phi=np.dot(H, psi)
    return 4*(np.sum(np.abs(phi)**2)-np.abs(np.sum(np.conjugate(phi)*psi))**2)



