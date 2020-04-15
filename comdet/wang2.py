#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:02:05 2019

@author: kanekoakihiro
"""

import numpy as np

def ichi(aa,bb):
    if aa == bb:
        return 1
    else:
        return 0

def ichi2(aa,bb,cc,dd):
    if aa==bb and cc==dd:
        return 1
    else:
        return 0

def likelihood(A,z):
    n=len(A)
    kp=list(set(z))
    #print(kp)
    klen=len(kp)
    #print(klen)
    naz=[0]*klen
    #print(naz)
    for k in range(klen):
        tmp=0
        for i in range(len(z)):
            tmp+=ichi(z[i],k)
        naz[k]=tmp
    #print(naz)
    
    nabz=[[0] * klen for i in range(klen)]
    #print(nabz)
    for l in range(klen):
        for k in range(klen):
            for i in range(len(z)):
                for j in range(len(z)):
                    if i!=j:
                        nabz[k][l]+=ichi2(z[i],k,z[j],l)

    Oabz=[[0] * klen for i in range(klen)]
    for l in range(klen):
        for k in range(klen):
            for i in range(len(z)):
                for j in range(len(z)):
                    if i!=j:
                        Oabz[k][l]+=A[i][j]*ichi2(z[i],k,z[j],l)
    #print(Oabz)

    pia=[0]*klen
    for i in  range(klen):
        pia[i]=naz[i]/n
    #print("Wang'spia",pia)

    Hab=[[0] * klen for i in range(klen)]
    for i in range(klen):
        for j in range(klen):
            Hab[i][j]=Oabz[i][j]/nabz[i][j]

    #print(Hab)

    """
    真の確率が0だったときの補正
    """
    for i in range(klen):
        for j in range(klen):
            if Hab[i][j]==0:
                Hab[i][j]=0.001

    #print("Wang'sHab",Hab)

    prefmlzka=1
    for i in range(klen):
        prefmlzka=prefmlzka*pia[i]**naz[i]

    #print(prefmlzka)
    postfmlzka=1
    for i in range(klen):
        for j in range(klen):
            postfmlzka=postfmlzka*Hab[i][j]**Oabz[i][j]*(1-Hab[i][j])**(nabz[i][j]-Oabz[i][j])

    postfmlzka=np.sqrt(postfmlzka)
    fmlzka=prefmlzka*postfmlzka
    #print(fmlzka)

    lambdalist=[0]*300

    for i in range(len(lambdalist)):
        lambdalist[i]=0.001*i

    beta=[0]*300
    for i in range(len(lambdalist)):
        beta[i]=np.log(fmlzka)-lambdalist[i]*klen*(klen+1)*n*np.log(n)/2

    return beta

def decide_com_num(betastack):
    betasum=[0]*300
    for i in range(300):
        for j in range(len(betastack)):
            betasum[i]=betasum[i]-betastack[j][i]

    #print(betasum)
    omegalist=[[0] * 300 for i in range(len(betastack))]
    #print(omegalist)
    for i in range(300):
        for j in range(len(betastack)):
            omegalist[j][i]=-betastack[j][i]/betasum[i]
 
    ent=[0] * 300     
    for i in range(300):
        for j in range(len(betastack)):
            ent[i]=ent[i]+omegalist[j][i]*np.log(omegalist[j][i])
        ent[i]=-ent[i]

    maxent=max(ent)
    mn=ent.index(maxent)

    entmaxbetalist=[0]*len(betastack)
    for i in range(len(betastack)):
        entmaxbetalist[i]=betastack[i][mn]
        
    print(entmaxbetalist)
    entmaxbeta=max(entmaxbetalist)
    #print(entmaxbetalist.index(entmaxbeta))
    return entmaxbetalist.index(entmaxbeta)

def likelihood2(A,z,pia,Hab):
    n=len(A)
    kp=list(set(z))
    #print(kp)
    klen=len(kp)
    #print(klen)
    naz=[0]*klen
    #print(naz)
    for k in range(klen):
        tmp=0
        for i in range(len(z)):
            tmp+=ichi(z[i],k)
        naz[k]=tmp
    #print(naz)
    
    nabz=[[0] * klen for i in range(klen)]
    #print(nabz)
    for l in range(klen):
        for k in range(klen):
            for i in range(len(z)):
                for j in range(len(z)):
                    if i!=j:
                        nabz[k][l]+=ichi2(z[i],k,z[j],l)

    Oabz=[[0] * klen for i in range(klen)]
    for l in range(klen):
        for k in range(klen):
            for i in range(len(z)):
                for j in range(len(z)):
                    if i!=j:
                        Oabz[k][l]+=A[i][j]*ichi2(z[i],k,z[j],l)
    #print(Oabz)

    #print(Hab)

    """
    真の確率が0だったときの補正
    """
    for i in range(klen):
        for j in range(klen):
            if Hab[i][j]==0:
                Hab[i][j]=0.001

    #print("Wang'sHab",Hab)

    prefmlzka=1
    for i in range(klen):
        prefmlzka=prefmlzka*pia[i]**naz[i]

    #print(prefmlzka)
    postfmlzka=1
    for i in range(klen):
        for j in range(klen):
            postfmlzka=postfmlzka*Hab[i][j]**Oabz[i][j]*(1-Hab[i][j])**(nabz[i][j]-Oabz[i][j])

    postfmlzka=np.sqrt(postfmlzka)
    fmlzka=prefmlzka*postfmlzka
    #print(fmlzka)

    lambdalist=[0]*300

    for i in range(len(lambdalist)):
        lambdalist[i]=0.001*i

    beta=[0]*300
    for i in range(len(lambdalist)):
        beta[i]=np.log(fmlzka)-lambdalist[i]*klen*(klen+1)*n*np.log(n)/2

    return beta

