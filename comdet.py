import numpy as np
import random
import functools
import operator
import math

def f(p):
    if random.random() < p:
        return 1
    else:
        return 0

def make_Ai(n,ph):
    Aitr=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            Aitr[i][j]=f(ph)
    return Aitr+Aitr.T

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

def decide_com_num(betastack):
    betasum=[0]*300
    for i in range(300):
        for j in range(len(betastack)):
            betasum[i]=betasum[i]-betastack[j][i]

    #print(betasum)
    omegalist=[[0] * 300 for i in range(4)]
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
        
    #print(entmaxbetalist)
    entmaxbeta=max(entmaxbetalist)
    #print(entmaxbetalist.index(entmaxbeta))
    return entmaxbetalist.index(entmaxbeta)

class Comdet:
    def __init__(self, A):
        self.A = A
    
    def plike(self, e):
        n = len(self.A)
        kl=list(set(e))
        klen=len(kl)
        
        for s in range(10):
            b = [[0] * klen for i in range(n)]
            for i in range(n):
                for k in range(klen):    
                    for j in range(n):
                        b[i][k]=b[i][k]+self.A[i][j]*ichi(e[j],kl[k])
            nk=[0] * klen
            
            for k in range(klen):
                for i in range(n):
                    nk[k]=nk[k]+ichi(e[i],kl[k])
            nkl=[[0] * klen for i in range(klen)]
    
            for k in range(klen):
                for l in range(klen):
                    if k!=l:
                        nkl[k][l]=nk[k]*nk[l]
                    else:
                        nkl[k][l]=nk[k]*(nk[k]-1)
            #print(nkl)
            Okl=[[0] * klen for i in range(klen)]
            
            for k in range(klen):
                for l in range(klen):
                    for i in range(n):
                        for j in range(n):
                            Okl[k][l]=Okl[k][l]+self.A[i][j]*ichi2(e[i],k,e[j],l)
            #print(Okl)  
            pihatl=[0] * klen
            for l in range(klen):
                pihatl[l]=nk[l]/n
                #print(pihatl)
            Rhat=np.diag(pihatl)
            #print(Rhat)
            Phatlk=[[0] * klen for i in range(klen)]
            for i in range(klen):
                for j in range(klen):
                    Phatlk[i][j]=Okl[i][j]/nkl[i][j]
            Phat=np.array(Phatlk)
            #print(Phat)
            lambdahatlk=[[0] * klen for i in range(klen)]
            for i in range(klen):
                for j in range(klen):
                    lambdahatlk[i][j]=n*np.dot(Rhat[i], Phat.T[j])
            #print(lambdahatlk)
    
            """
            ここから下を繰り返す.
            """
            N2=10
            count2=1
            while True:
                pihatil=[[0] * n for i in range(klen)]
                for l in range(klen):
                    for i in range(n):
                        prodlistl=[]
                        for m in range(klen):
                            if lambdahatlk[l][m] == 0:
                                #print("Stop")
                                break
                        
                            prodlistl.append(math.exp(b[i][m]*math.log(lambdahatlk[l][m])-lambdahatlk[l][m]))
                            bunsi = pihatl[l]*functools.reduce(operator.mul, prodlistl)
                        bunbo=0
                        for k in range(klen):
                            prodlistk=[]
                            for m in range(klen):
                                if lambdahatlk[k][m] == 0:
                                    #print("Stop")
                                    break
                                prodlistk.append(math.exp(b[i][m]*math.log(lambdahatlk[k][m])-lambdahatlk[k][m]))
                            #print(functools.reduce(operator.mul, prodlistk))
                            plik = functools.reduce(operator.mul, prodlistk)
                            bunbo=bunbo+pihatl[k]*plik
                        pihatil[l][i]=bunsi/bunbo
    
                pihatl=[0] * klen
                for l in range(klen):
                    for i in range(n):
                        pihatl[l]=pihatl[l]+pihatil[l][i]
    
                lambdahatlk=[[0] * klen for i in range(klen)]
                for l in range(klen):
                    for k in range(klen):
                        pisum=0
                        for i in range(n):
                            lambdahatlk[k][l]=lambdahatlk[k][l]+b[i][k]*pihatil[l][i]
                            pisum=pisum+pihatil[l][i]
                        lambdahatlk[k][l]=lambdahatlk[k][l]/pisum
                count2 += 1
                if count2>N2:
                    break
            
            maxlist=[0] * n
            pihatilt=np.array(pihatil).T
    
            for i in range(i):
                maxlist[i]=max(pihatilt[i])
                #print(maxlist)
            for i in range(n):
                e[i]=np.argmax(pihatilt[i])
            #print(e)
        
        return e
    
    def devG(self,e):
        kl=list(set(e))
        klen=len(kl)
        etl=[]
        for j in range(klen):
            etl.append([i for i, x in enumerate(e) if x == j])

        A_res=[]
        for k in range(klen):
            A1=[]
            for i in etl[k]:
                for j in etl[k]:
                    A1.append(self.A[i][j])
            arr_A1=np.array(A1)
            A2=arr_A1.reshape([len(etl[k]),len(etl[k])])
            A_res.append(A2)
        return A_res
    
    def likelihood(self,z):
        n=len(self.A)
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
                            Oabz[k][l]+=self.A[i][j]*ichi2(z[i],k,z[j],l)
        #print(Oabz)

        pia=[0]*klen
        for i in  range(klen):
            pia[i]=naz[i]/n
        #print(pia)

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

        #print(Hab)

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
        
if __name__ == "__main__": 
    Karate=[[0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0,
   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1,
   1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
  0], [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
  1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 1, 0, 
  0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0], [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0,
   0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1,
   1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
   0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
  0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 1,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 
  0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 
  1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
   0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
  0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 
  0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 0]]
    A=Comdet(Karate)
    ee=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    et=A.plike(ee)
    print('クラス分けは',et)
    #print(A.likelihood(A.plike(ee)))
    zz1=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    zz2=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    zz3=[0, 0, 0, 0, 2, 2, 2, 0, 1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    l1=A.likelihood(zz1)
    l2=A.likelihood(zz2)
    l3=A.likelihood(zz3)
    betastack=np.stack([l1, l2, l3])
    K=decide_com_num(betastack)

    print('最適なクラスタ数は',K)
    
