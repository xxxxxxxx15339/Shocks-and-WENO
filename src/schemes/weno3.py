# -*- coding: utf-8 -*- 
"""
Created on Satu 11 July 16:37:00 2026 

@author: yasser ba 
Inspired by the work of the original author: ben91
"""
from SimulationClasses import * 
import numpy as np 

def WENO3():
    def scheme(u):
        """
        vi+1/2 = v(xi+1/2) + O(Dx^2k-1)
        for k = 2: vi+1/2 = v(xi+1/2) + O(Dx^3) 3th order accuracy

        """
        ep = 1E-6 
    
        # Fluxes on sub stencils
        f1 =-1/2*u[:,0]+3/2*u[:,1]
        f2 = 1/2*u[:,1]+1/2*u[:,2]
            
        # Smoothness Inidicators: 
        B1 = np.power(u[:,1] - u[:,0], 2)
        B2 = np.power(u[:,2] - u[:,1], 2) 

        # Linear Weights: 
        g1 = 1/3 
        g2 = 2/3 

        # Unscaled Nonlinear weights: 
        wt1 = g1/np.power(ep+B1, 2)
        wt2 = g2/np.power(ep+B2, 2)
        wts = wt1 + wt2

        # Scaled Nonlinear weights:
        w1 = wt1/wts
        w2 = wt2/wts

        # Compute the flux: 
        fl = f1*w1 + f2*w2
        return fl 
    FVM = FiniteVolumeMethod(3, scheme)
    return FVM 

def NNMethod(model):    
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,1])
        u_tmp[:] = u[:,1]
        for i in range(0,3): 
            denominator = max_u - min_u
            safe_denominator = np.where(denominator == 0, 1.0, denominator)
            u[:,i] = np.where(denominator != 0, (u[:,i] - min_u) / safe_denominator, 0.0)

        fl = model.predict(u)#compute \Delta u
        fl = fl.flatten()
        fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethod(3, scheme)
    return FVM

def NNMethod_noScale(model):    
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,1])
        u_tmp[:] = u[:,1]
        #for i in range(0,5):
            #u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        fl = model.predict(u)#compute \Delta u
        fl = fl.flatten()
        #fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethod(3, scheme)
    return FVM


