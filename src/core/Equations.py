# -*- coding: utf-8 -*- 
"""
Created on Wed 8 July 03:21:00 2026 

@author: yasser ba 
Inspired by the work of the original author: ben91
"""

import numpy as np 

# Two main equations that we will be using: advection - Burger  

def adv():  
    # Returns a trivial flux function where f(u) = u 
    def flux(u):
        return u
    
    return flux

def invBurg():
    # Defines a none linear PDE.
    def flux(u):
        return 0.5*np.power(u,2)
    return flux

def euler():
    from .eulerEquations import flux as euler_flux
    return euler_flux



