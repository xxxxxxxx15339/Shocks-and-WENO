# -*- coding: utf-8 -*- 
"""
Created on Thur 9 July 13:45:00 2026 

@author: yasser ba 
Inspired by the work of the original author: ben91
"""
from SimulationClasses import * 
import numpy as np 

def WENO5():
    def scheme():
        """
        vi+1/2 = v(xi+1/2) + O(Dx^2k-1)
        for k = 3: vi+1/2 = v(xi+1/2) + O(Dx^5) 5th order accuracy

        """
        ep = 1E-6
    
        # Fluxes on sub stencils
        f1 = 1/3*u[:,0]-7/6*u[:,1]+11/6*u[:,2]
        f2 = -1/6*u[:,1]+5/6*u[:,2]+1/3*u[:,3]
        f3 = 1/3*u[:,2]+5/6*u[:,3]-1/6*u[:,4]
            
        # Smoothness Inidicators: 
        B1 = 13/12*np.power(u[:,0]-2*u[:,1]+u[:,2],2) + 1/4*np.power(u[:,0]-4*u[:,1]+3*u[:,2],2)
        B2 = 13/12*np.power(u[:,1]-2*u[:,2]+u[:,3],2) + 1/4*np.power(u[:,1]-u[:,3],2)
        B3 = 13/12*np.power(u[:,2]-2*u[:,3]+u[:,4],2) + 1/4*np.power(3*u[:,2]-4*u[:,3]+u[:,4],2)

        # Linear Weights: 
        g1 = 1/10 
        g2 = 3/5 
        g3 = 3/10 

        # Unscaled Nonlinear weights: 
        wt1 = g1/np.power(ep+B1, 2)
        wt2 = g2/np.power(ep+B2, 2)
        wt3 = g3/np.power(ep+B3, 2)
        wts = wt1 + wt2 + wt3

        # Scaled Nonlinear weights:
        w1 = wt1/wts
        w2 = wt2/wts
        w3 = wt3/wts

        # Compute the flux: 
        fl = f1*w1 + f2*w2 + f3*w3 
        return fl 
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM 



