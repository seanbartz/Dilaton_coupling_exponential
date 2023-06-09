#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDITED on November 15 2022:
    Parallel processing with Pools for speedup by factor of ~3 on an 8-CPU machine

EDITED ON June 9 2022:
    Added mixing term between dilaton and chiral field
    
Created on Tue March 16  2021
For a given quark mass and chemical potential, 
solves for all sigma values for a range of temperatures.
If there are multiple values, then the transition is 1st order.
@author: seanbartz
"""
import numpy as np
# import math
from scipy.integrate import odeint
from solveTmu import blackness

from timebudget import timebudget

import matplotlib.pyplot as plt
from multiprocessing import Pool
import os


# import time




# start_time=time.perf_counter()

def chiral(y,u,params):
    chi,chip=y
    v3,v4,lambda1,mu0,mu1,mu2,zh,q=params
    
    Q=q*zh**3
    
    
    "Exponential parameterization"
    phi = -(mu1*zh*u)**2+(mu0**2+mu1**2)*(zh**2)*(u**2)*(1-np.exp(-(mu2**2)*(zh**2)*(u**2)))
    phip = -2*u*(mu1*zh)**2+2*(mu1**2+mu0**2)*(mu**2)*(zh**4)*(u**3)*np.exp(-(mu2**2)*(zh**2)*(u**2))+2*(mu1**2+mu0**2)*(zh**2)*u*(1-np.exp(-(mu2**2)*(zh**2)*(u**2)))
                                                    
    f= 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp= -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    "EOM for chiral field"
    derivs=[chip,
            (3/u-fp/f+phip)*chip - (3*chi+lambda1*phi*chi-3*v3*chi**2-4*v4*chi**3)/(u**2*f)]
            #((3+u**4)/(u-u**5) +phip)*chip - (-3*chi+4*v4*chi**3)/(u**2-u**6) ]
            
    return derivs
# @timebudget
def allSigmas(args):#,mu,ml,minsigma,maxsigma,a0,lambda1):
    "Unpack the input"
    T,mu,ml,minsigma,maxsigma,a0,lambda1=args

    minsigma=int(minsigma)
    maxsigma=int(maxsigma)
    deltasig = 1
    sigmavalues = np.arange(minsigma,maxsigma,deltasig)
    mu0 = 430
    mu1 = 830
    mu2 = 176
    mu_g = 440
    "solve for horizon and charge"
    zh,q=blackness(T,mu)
    Q=q*zh**3
    """
    limits of spatial variable z/zh. Should be close to 0 and 1, but 
    cannot go all the way to 0 or 1 because functions diverge there
    """
    ui = 0.01
    uf = 0.999
    "Create the spatial variable mesh"
    umesh=100
    u=np.linspace(ui,uf,umesh)
    

    
    
 
    
    "This is a constant that goes into the boundary conditions"
    zeta=np.sqrt(3)/(2*np.pi)
    
    "For the scalar potential in the action"
    "see papers by Bartz, Jacobson"
    #v3= -3 #only needed for 2+1 flavor
    # v4 = 8
    # v3 = -3
    
    "Matching Fang paper"
    v4=4.2
    v3= -22.6/(6*np.sqrt(2))
    
    "need the dilaton for mixing term in test function"
    "Exponential parameterization"
    phi = -(mu1*zh*u)**2+(mu0**2+mu1**2)*(zh**2)*(u**2)*(1-np.exp(-(mu2**2)*(zh**2)*(u**2)))
        
    #sigmal=260**3
    params=v3,v4,lambda1,mu0,mu1,mu2,a0,zh,q
    "blackness function and its derivative, Reissner-Nordstrom metric"
    "This version is for finite temp, finite chemical potential"
    f = 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp = -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    
    "stepsize for search over sigma"
    "Note: search should be done over cube root of sigma, here called sl"
    #tic = time.perf_counter()

    truesigma = 0
    "This version steps over all values to find multiple solutions at some temps"
    
    "initial values for comparing test function"
    oldtest=0
    j=0
    truesigma=np.zeros(3)
    
    
    s2=-3*(ml*zeta)**2*v3
    s3=-9*(zeta*ml)**3*v3**2 + 2*(zeta*ml)**3*v4 + ml*zeta*mu_g**2 - 1/2*ml*zeta*lambda1*mu_g**2

    #for sl in range (minsigma,maxsigma,deltasig):
    for i in range(len(sigmavalues)):
        sl=sigmavalues[i]
        "values for chiral field and derivative at UV boundary"
        sigmal = sl**3
        UVbound = [ml*zeta*zh*ui + sigmal/zeta*(zh*ui)**3+s2*(zh*ui)**2+s3*(zh*ui)**3*np.log(zh*ui), 
                   ml*zeta*zh + 3*sigmal/zeta*zh**3*ui**2 + 2*s2*zh**2*ui + s3* ui**2*zh**3*(1+3*np.log(zh*ui))]
           
        "solve for the chiral field"
        chiFields=odeint(chiral,UVbound,u,args=(params,))
        
        "test function defined to find when the chiral field doesn't diverge"
        "When test function is zero at uf, the chiral field doesn't diverge"
        test = ((-u**2*fp)/f)*chiFields[:,1]-1/f*(3*chiFields[:,0]+lambda1*phi*chiFields[:,0]-3*v3*chiFields[:,0]**2-4*v4*chiFields[:,0]**3)
        testIR = test[umesh-1]#value of test function at uf
        
        "when test function crosses zero, it will go from + to -, or vice versa"
        "This is checked by multiplying by value from previous value of sigma"
        if oldtest*testIR<0: #and chiFields[umesh-1,0]>0:
           
            truesigma[j]=sl #save this value
            j=j+1 #if there are other sigma values, they will be stored also
            #print(truesigma)
        if j>2:
            break
            
        oldtest=testIR

    
    return truesigma

@timebudget
def get_all_sigmas(operation, input):
    "This function executes a loop to calculate all sigma values for all values of the temps array"
    truesigma=np.zeros([len(input),3])

    for i in range(0,len(input)):
        truesigma[i,:]=operation(input[i])#,100,24,0,300,0,7.438)
    return truesigma

@timebudget
def get_all_sigmas_parallel(operation,input,pool):
    truesigma=np.zeros([len(input),3])

    truesigma=pool.map(operation, input)
    
    return truesigma
    

if __name__ == '__main__':

        
    tmin=0
    tmax=100
    numtemp=50
    
    
    temps=np.linspace(tmin,tmax,numtemp)
    
    #light quark mass
    ml=24*np.ones(numtemp)
    
    #chemical potential
    mu=800*np.ones(numtemp)
    
    lambda1=5*np.ones(numtemp) #parameter for mixing between dilaton and chiral field
    
    minsigma=0*np.ones(numtemp)
    maxsigma=500*np.ones(numtemp)
    
    a0=0.*np.ones(numtemp)
    
    
    tempsArgs=np.array([temps,mu,ml,minsigma,maxsigma,a0,lambda1]).T


    #need up to 3 sigma values per temperature
    #truesigma=np.zeros([numtemp,3])
    
    "This calls the old version, which loops over all temps. Only un-comment for speed comparisons"
    #truesigma=get_all_sigmas(allSigmas,tempsArgs)
    
    "Create a pool that uses all available cpus"
    processes_count=os.cpu_count()    
    processes_pool = Pool(processes_count)
    
    #Produces: TypeError: only size-1 arrays can be converted to Python scalars
    truesigma=get_all_sigmas_parallel(allSigmas,tempsArgs,processes_pool)
    truesigma=np.array(truesigma)
    processes_pool.close()
    
        
    plt.scatter(temps,truesigma[:,0])
    plt.scatter(temps,truesigma[:,1])
    plt.scatter(temps,truesigma[:,2])
    plt.ylim([min(truesigma[:,0])-5,max(truesigma[:,0])+5])
    plt.xlabel('Temperature (MeV)')
    plt.ylabel(r'$\sigma^{1/3}$ (MeV)')
    plt.title(r'$m_q=%i$ MeV, $\mu=%i$ MeV, $\lambda_1=$ %f' %(ml[0],mu[0],lambda1[0]))
    plt.show()

    if max(truesigma[:,1])==0:
        print("Crossover or 2nd order")
        #find the temp value where the gradient of truesigma[:,0] is most negative
        #this is the pseudo-critical temperature
        print("Pseudo-Critical temperature is between", temps[np.argmin(np.gradient(truesigma[:,0]))-1], temps[np.argmin(np.gradient(truesigma[:,0]))] )
        #these temperature values are the new bounds for the next iteration
        tmin=temps[np.argmin(np.gradient(truesigma[:,0]))-1]
        tmax=temps[np.argmin(np.gradient(truesigma[:,0]))]

        #these values of sigma are the new bounds for the next iteration
        maxsigma=truesigma[np.argmin(np.gradient(truesigma[:,0]))-1,0]
        minsigma=truesigma[np.argmin(np.gradient(truesigma[:,0])),0]

        #print the sigma values for the new bounds
        print("Sigma bounds for the next search are ", minsigma, maxsigma)
    else:
        print("First order")  
        #crtical temperature is where truesigma has multiple solutions  
        print("Critical temperature is ", temps[np.argmax(truesigma[:,1])] )
    # end_time=time.perf_counter()
    # print("Time elapsed = ", end_time-start_time )