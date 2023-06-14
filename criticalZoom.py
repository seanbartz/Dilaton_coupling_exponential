#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDITED On June 12 2023:
    Only make deltasig = 0.1 if the range of sigma is <= 1. Otherwise, spurious results can be found

EDITED On June 2 2023: 
    Looks for minimum of gradient of sigma instead of sigma^(1/3), which is more reliable,
    and will not accidentally say the transition temperature occurs when sigma goes to zero..

    Because of this, it is no longer necessary to remove the zero values of sigma,
    which allows the maximum value of T to be arbitrarily large. 
    This makes the code more robust, and allows for a more accurate determination of the critical temperature.

    Also, removed any points from truesigma[:,1] and truesigma[:,2] that are greater than 
    the maximum value of truesigma[:,0], which helps prevent spurious results.
    We don't need to fine tune the value of maxsiga or tmin as much.

Created on Friday December 30 2022
Split from threeflavorALLvalues.py

This version iteratively zooms in on the critical temperature, terminating when a first order transition is found.


Notes from previous version:
-----------------------
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
from scipy.integrate import odeint
from solveTmu import blackness

from timebudget import timebudget

import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import pandas as pd 


# import time




# start_time=time.perf_counter()

def chiral(y,u,params):
    chi,chip=y
    v3,v4,lambda1,mu0,mu1,mu2,zh,q=params
    
    Q=q*zh**3
    
    
    "Exponential parameterization"
    phi = -(mu1*zh*u)**2+(mu0**2+mu1**2)*(zh**2)*(u**2)*(1-np.exp(-(mu2**2)*(zh**2)*(u**2)))
    phip = 2 * u * zh**2 * (mu0**2 + np.exp(-u**2 * zh**2 * mu2**2) * (mu0**2 + mu1**2) * (-1 + u**2 * zh**2 * mu2**2))
                     

    f= 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp= -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    "EOM for chiral field"
    derivs=[chip,
            (3/u-fp/f+phip)*chip - (3*chi+lambda1*phi*chi-3*v3*chi**2-4*v4*chi**3)/(u**2*f)]
            #((3+u**4)/(u-u**5) +phip)*chip - (-3*chi+4*v4*chi**3)/(u**2-u**6) ]
            
    return derivs
# @timebudget
def allSigmas(args):
    "Unpack the input"
    T,mu,ml,minsigma,maxsigma,lambda1,mu0,mu1,mu2=args

    minsigma=int(minsigma)
    maxsigma=int(maxsigma)
    

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
    v4 = 8
    v3 = -3
    
    # "Matching Fang paper"
    # v4=4.2
    # v3= -22.6/(6*np.sqrt(2))
    
    "need the dilaton for mixing term in test function"
    "Exponential parameterization"
    phi = -(mu1*zh*u)**2+(mu0**2+mu1**2)*(zh**2)*(u**2)*(1-np.exp(-(mu2**2)*(zh**2)*(u**2)))
        
    #sigmal=260**3
    params=v3,v4,lambda1,mu0,mu1,mu2,zh,q
    "blackness function and its derivative, Reissner-Nordstrom metric"
    "This version is for finite temp, finite chemical potential"
    f = 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp = -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    
    "stepsize for search over sigma"
    "Note: search should be done over cube root of sigma, here called sl"
    if maxsigma-minsigma<=1:
        deltasig = 0.1
    else:
        deltasig = 1
    #tic = time.perf_counter()

    # create an array of sigma values from minsigma to maxsigma, incrementing by deltasig
    sigmavalues = np.arange(minsigma,maxsigma,deltasig)

    truesigma = 0
    "This version steps over all values to find multiple solutions at some temps"
    
    "initial values for comparing test function"
    oldtest=0
    j=0
    truesigma=np.zeros(3)
    
    
    s2 = -3*(ml*zeta)**2*v3
    s3 = -9*(zeta*ml)**3*v3**2 + 2*(zeta*ml)**3*v4 - ml*zeta*mu1**2 + 1/2*ml*zeta*lambda1*mu1**2


    #for sl in range (minsigma,maxsigma,deltasig):
    for i in range(len(sigmavalues)):
        sl=sigmavalues[i]
        "values for chiral field and derivative at UV boundary"
        sigmal = sl**3
        s4 = 9 * ml**4 * v3**3 * zeta**4 - 12 * ml**4 * v3 * v4 * zeta**4 + 4 * ml**2 * v3 * zeta**2 * mu1**2 - ml**2 * v3 * zeta**2 * lambda1 * mu1**2 + 2 * ml * v3 * sigmal

        UVbound = [ml*zeta*zh*ui + sigmal/zeta*(zh*ui)**3+s2*(zh*ui)**2+s3*(zh*ui)**3*np.log(zh*ui) + s4*(zh*ui)**4, 
                   ml*zeta*zh + 3*sigmal/zeta*zh**3*ui**2 + 2*s2*zh**2*ui + s3* ui**2*zh**3*(1+3*np.log(zh*ui)) + 4*s4*zh**4*ui**3]
           
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

'''
 this function finds all sigma values for a range of temperatures. 
 Its input is a range of temperatures, the number of temperature values, and the sigma range.
 It outputs whether the transition is first order or second order.
 If second order, it outputs new bounds for the sigma values and the new temperature range.
'''
def order_checker(tmin,tmax,numtemp,minsigma,maxsigma, ml, mu, lambda1,mu0,mu1,mu2):

    temps=np.linspace(tmin,tmax,numtemp)
    lambda1 = lambda1*np.ones(numtemp)
    ml = ml*np.ones(numtemp)
    mu = mu*np.ones(numtemp)
    mu0 = mu0*np.ones(numtemp)
    mu1 = mu1*np.ones(numtemp)
    mu2 = mu2*np.ones(numtemp)


    minsigma=minsigma*np.ones(numtemp)
    maxsigma=maxsigma*np.ones(numtemp)

    tempsArgs=np.array([temps,mu,ml,minsigma,maxsigma,lambda1,mu0,mu1,mu2]).T


    #need up to 3 sigma values per temperature
    # truesigma=np.zeros([numtemp,3])
    
    "This calls the old version, which loops over all temps. Only un-comment for speed comparisons"
    # truesigma=get_all_sigmas(allSigmas,tempsArgs)
    
    "Create a pool that uses all available cpus"
    processes_count=os.cpu_count()    
    processes_pool = Pool(processes_count)
    
    truesigma=get_all_sigmas_parallel(allSigmas,tempsArgs,processes_pool)
    truesigma=np.array(truesigma)
    processes_pool.close()
    
        
    #if any values  of truesigma[:,1] or truesigma[:,2] are greater than the maximum value of truesigma[:,0], set them to zero
    #these points are spurious.
    truesigma[truesigma[:,1]>max(truesigma[:,0]),1]=0
    truesigma[truesigma[:,2]>max(truesigma[:,0]),2]=0

    
    if max(truesigma[:,1])==0:
        print("Crossover or 2nd order")
        #keep only the non-zero values of truesigma, and the corresponding temperatures
#         temps=temps[truesigma[:,0]!=0]
#         truesigma=truesigma[truesigma[:,0]!=0]

        numtemp=len(temps)
        #find the temp value where the gradient of truesigma[:,0]**3 is most negative, and the value of truesigma[:,0] is not zero
        #this is the pseudo-critical temperature

        #NOTE: we get better results when looking at the gradient of truesigma[:,0]**3, rather than truesigma[:,0]
        # this avoids accidentally identifying the temperature at which truesigma[:,0] goes to zero as the critical temperature
        transitionIndex=np.argmin(np.gradient(truesigma[:,0]**3))
        Tc=temps[transitionIndex]
        buffer=2
        print("Pseudo-Critical temperature is between", temps[max(transitionIndex-buffer,0)], temps[min(transitionIndex+buffer,numtemp-1)] )
        #these temperature values are the new bounds for the next iteration, with a buffer of 1
        tmin=temps[max(transitionIndex-buffer,0)]
        tmax=temps[min(transitionIndex+buffer,numtemp-1)]

        #these values of sigma are the new bounds for the next iteration
        maxsigma=truesigma[0,0]+1
        minsigma=truesigma[numtemp-1,0]

        #print the sigma values for the new bounds
        # print("Sigma bounds for the next search are ", minsigma, maxsigma)
        order=2
    else:
        print("First order")  
        #crtical temperature is where truesigma has multiple solutions  
        Tc= temps[np.argmax(truesigma[:,1])]
        print("Critical temperature is ", Tc )


        order=1
    return tmin,tmax,minsigma,maxsigma,order,temps,truesigma,Tc

@timebudget
def critical_zoom(tmin,tmax,numtemp,minsigma,maxsigma,ml,mu,lambda1,mu0,mu1,mu2):
    order=2
    iterationNumber=0

    #create a list to store the sigma values, temperatures, and order of the transition
    sigma_list=[]
    temps_list=[]
    Tc=tmin

    #iteratively run the order_checker function until the transition is first order, or until the bounds are too small
    while order==2 and iterationNumber<10 and tmin<tmax and maxsigma-minsigma>2 and (tmax-tmin)/numtemp>1.0e-4:
        tmin,tmax,minsigma,maxsigma,order,temps,truesigma,Tc=order_checker(tmin,tmax,numtemp,minsigma,maxsigma,ml,mu,lambda1,mu0,mu1,mu2)
        iterationNumber=iterationNumber+1
        print("Iteration number ", iterationNumber)
        if tmax<tmin:
            print("TEMPERATURE BOUNDS REVERSED!!!")
        sigma_list.append(truesigma)
        temps_list.append(temps)

    return order, iterationNumber, sigma_list,temps_list,Tc


if __name__ == '__main__':

    #temperature range
    tmin=80
    tmax=210
    # number of temperature values
    numtemp=25

    #light quark mass
    ml=24

    #chemical potential
    mu=150

    lambda1= 7.8 #parameter for mixing between dilaton and chiral field

    #sigma range
    minsigma=0
    maxsigma=400

    mu0=430
    mu1=830
    mu2=176

    order, iterationNumber, sigma_list,temps_list,Tc=critical_zoom(tmin,tmax,numtemp,minsigma,maxsigma,ml,mu,lambda1,mu0,mu1,mu2)



    #plot all the sigma values for each iteration
    #get the standard colors for matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    #find the index of when sigma_list[0][:,0] has its first zero value
    #this is the index of the first temperature where the sigma value is zero

    max_index=np.argmax(sigma_list[0][:,0]==0)
    #find the value of the temperature at this index
    max_temp=temps_list[0][max_index]



    for i in range(len(sigma_list)):
        plt.scatter(temps_list[i],(sigma_list[i][:,0]/1000)**3,color=colors[0])
        plt.scatter(temps_list[i],(sigma_list[i][:,1]/1000)**3,color=colors[1])
        plt.scatter(temps_list[i],(sigma_list[i][:,2]/1000)**3,color=colors[2])
    plt.xlabel("Temperature (MeV)")
    plt.ylabel("$\sigma$ (GeV)$^3$")
    #set the x range
    plt.xlim(temps_list[0][0],max_temp)
    plt.title(r'$m_q=%i$ MeV, $\mu=%i$ MeV, $\lambda_1=$ %f' %(ml,mu,lambda1))
    plt.show()
        
    
    # Save the data as a pandas data frame
    df_all_list = []
    for i in range(len(sigma_list)):
        df=pd.DataFrame()
        df['temps']=temps_list[i]
        df['sigma1']=(sigma_list[i][:,0]/1000)**3   
        df['sigma2']=(sigma_list[i][:,1]/1000)**3
        df['sigma3']=(sigma_list[i][:,2]/1000)**3
        df['order']=order
        df['ml']=ml
        df['mu']=mu
        df['lambda1']=lambda1
        df['Tc']=Tc
        df_all_list.append(df)

    # Use pandas concat here
    df_all = pd.concat(df_all_list)

    #pickle the data frame
    df_all.to_pickle('data/chiral_transition_mq%i_mu%i_lambda1%f_order%i.pkl' %(ml,mu,lambda1,order))
