# -*- coding: utf-8 -*- 
"""
Created on Satu 11 July 16:37:00 2026 

@author: yasser ba 
Inspired by the work of the original author: ben91
"""
# src/schemes/ENO3.py, Hook5.py, weno3.py, weno5.py, weno7.py
from ..core.SimulationClasses import FiniteVolumeMethod, FiniteVolumeMethodEuler
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
        B1 = (u[:,1] - u[:,0])**2
        B2 = (u[:,2] - u[:,1])**2 

        # Linear Weights: 
        g1 = 1/3 
        g2 = 2/3 

        # Unscaled Nonlinear weights: 
        wt1 = g1/(ep+B1)**2
        wt2 = g2/(ep+B2)**2
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

def WENO3euler():
    def scheme(f):
        '''
        inputs:
            f: variables to WENO5 (nx3x5)
        outputs:
            fl: WENO5 flux
        '''

        ep = 1E-6 
    
        # Fluxes on sub stencils
        f1 =-1/2*f[:,:,0]+3/2*f[:,:,1]
        f2 = 1/2*f[:,:,1]+1/2*f[:,:,2]
            
        # Smoothness Inidicators: 
        B1 = (f[:,:,1] - f[:,:,0])**2
        B2 = (f[:,:,2] - f[:,:,1])**2 

        # Linear Weights: 
        g1 = 1/3 
        g2 = 2/3 

        # Unscaled Nonlinear weights: 
        wt1 = g1/(ep+B1)**2
        wt2 = g2/(ep+B2)**2
        wts = wt1 + wt2

        # Scaled Nonlinear weights:
        w1 = wt1/wts
        w2 = wt2/wts

        # Compute the flux: 
        fl = f1*w1 + f2*w2
        return fl
    FVM = FiniteVolumeMethodEuler(3, scheme)
    return FVM

def NNEuler(model):    
    def scheme(u_all):
        n,m,s = np.shape(u_all)
        fl = np.zeros((n,m))
        u = np.zeros((n,s))
        for i in range(0,m):
            u[:,:] = u_all[:,i,:]
            min_u = np.amin(u,1)
            max_u = np.amax(u,1)
            const_n = min_u==max_u
            #print('u: ', u)
            u_tmp = np.zeros_like(u[:,1])
            u_tmp[:] = u[:,1]
            for j in range(0,3):
                denominator = max_u - min_u
                safe_denominator = np.where(denominator == 0, 1.0, denominator)
                u[:,j] = np.where(denominator != 0, (u[:,j] - min_u) / safe_denominator, 0.0)

            fl[:,i] = model.predict(u).flatten()#compute \Delta u
            #fl = fl.flatten()
            fl[:,i] = np.multiply(fl[:,i],(max_u-min_u))+min_u
            fl[const_n,i] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethodEuler(3, scheme)
    return FVM
