#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:30:13 2021
Solves for zh and q given the temperature and quark chemical potential
Call blackness to solve given
@author: seanbartz
"""

# from scipy.optimize import fsolve
import math




# def relations(vars,params):
#     T,mu=params
#     kappa=1
#     zh,q=vars
#     "Based on blackness function f(z) for Reissner-Nordstrom black hole"
#     eq1 = T-1/(math.pi*zh)*(1-q**2*zh**6/2)
#     eq2 = mu-kappa*q*zh**2
#     return [eq1,eq2]
def blackness(T,mu):
    kappa=1
    
    if mu==0:
        zh=1/(math.pi*T)
        q=0
    else:
        zh=kappa/mu**2*(-kappa*math.pi*T+math.sqrt((kappa*math.pi*T)**2+2*mu**2))
        q=(mu**3+kappa*math.pi*T*mu*(kappa*math.pi*T+math.sqrt((kappa*math.pi*T)**2+2*mu**2)))/(2*kappa**3)
        
    # inputs=(temp,chemPotential)
    # zh,q=fsolve(relations,args=(inputs,),x0=[1/(math.pi*150),1])
    return zh,q
