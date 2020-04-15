#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:48:37 2019

@author: kanekoakihiro
"""

from statistics import mean,stdev
from comdet import tw
import numpy as np
import random

"""
Algorithm 1
"""

def Hypothesis(A):
    TW=tw.TracyWidom(beta=1)
    n=len(A)
    I=np.eye(n)
    eet=1/n*np.ones((n,n,))
    sum=0
    for i in range(n):
        for j in range(n):
            sum+=A[i][j]
    phat=sum/(n*(n-1))
    if phat==0:
        print('no  edge in the graph')
    elif phat==1:
        print('これは完全グラフです')
    Phat=n*phat*eet-phat*I
    Apt=(A-Phat)/(np.sqrt((n-1)*phat*(1-phat)))
    la,v=np.linalg.eig(Apt)
    la1=max(la)
    la1Apt=(la1-2.0)*n**(2/3)
    return 1-TW.cdf(la1Apt)

"""
Algorithm 2
"""

def Hypothesis2(A):
    TW=tw.TracyWidom(beta=1)
    listtheta=[]
    n=len(A)
    I=np.eye(n)
    eet=1/n*np.ones((n,n,))
    sum=0
    for i in range(n):
        for j in range(n):
            sum+=A[i][j]
    phat=sum/(n*(n-1))
    if phat==0:
        print('no  edge in the graph')
    elif phat==1:
        print('これは完全グラフです')
    Phat=n*phat*eet-phat*I
    Apt=(A-Phat)/(np.sqrt((n-1)*phat*(1-phat)))
    la,v=np.linalg.eig(Apt)
    la1=max(la)
    theta=(la1-2.0)*n**(2/3)
    """
    Mathematicaの数字でTracy-Widom分布の平均分散を使う
    """
    meantw=-1.20653
    sdtw=1.26798
    for i in range(50):
        Ai=make_Ai(n,phat)
        sum=0
        for i in range(n):
            for j in range(n):
                sum+=Ai[i][j]
                
        pihat=sum/(n*(n-1))
        Pihat=n*pihat*eet-pihat*I
        Aipt=(Ai-Pihat)/(np.sqrt((n-1)*pihat*(1-pihat)))
        lai,vi=np.linalg.eig(Aipt)
        lai1=max(lai)
        lai1=lai1.real
        thetai=(lai1-2.0)*n**(2/3)
        listtheta.append(thetai)
    muhat=mean(listtheta)
    sigmahat=stdev(listtheta)
    thetap=meantw+((theta-muhat)/sigmahat)*sdtw
    return 1-TW.cdf(thetap)

def make_Ai(n,ph):
    Aitr=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            Aitr[i][j]=f(ph)
    return Aitr+Aitr.T
    

def f(p):
    if random.random() < p:
        return 1
    else:
        return 0