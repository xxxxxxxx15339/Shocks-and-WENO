# -*- coding: utf-8 -*- 
"""
Created on Thur 9 July 13:45:00 2026 

@author: yasser ba 
Inspired by the work of the original author: ben91
"""
# src/schemes/ENO3.py, Hook5.py, weno3.py, weno5.py, weno7.py
from ..core.SimulationClasses import *
import numpy as np 


def WENO5():
    def scheme(u):
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
        B1 = 13/12*(u[:,0]-2*u[:,1]+u[:,2])**2 + 1/4*(u[:,0]-4*u[:,1]+3*u[:,2])**2
        B2 = 13/12*(u[:,1]-2*u[:,2]+u[:,3])**2 + 1/4*(u[:,1]-u[:,3])**2
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

def NNMethod(model):    
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        for i in range(0,5): 
            denominator = max_u - min_u
            safe_denominator = np.where(denominator == 0, 1.0, denominator)
            u[:,i] = np.where(denominator != 0, (u[:,i] - min_u) / safe_denominator, 0.0)

        fl = model.predict(u)#compute \Delta u
        fl = fl.flatten()
        fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM

def NNMethod_noScale(model):    
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        #for i in range(0,5):
            #u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        fl = model.predict(u)#compute \Delta u
        fl = fl.flatten()
        #fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM


def WENO5euler():
    def scheme(f):
        '''
        inputs:
            f: variables to WENO5 (nx3x5)
        outputs:
            fl: WENO5 flux
        '''
        ep = 1E-6
        #compute fluxes on sub stencils
        f1 = 1/3*f[:,:,0]-7/6*f[:,:,1]+11/6*f[:,:,2]
        f2 =-1/6*f[:,:,1]+5/6*f[:,:,2]+ 1/3*f[:,:,3]
        f3 = 1/3*f[:,:,2]+5/6*f[:,:,3]- 1/6*f[:,:,4]
        #compute smoothness indicators
        B1 = 13/12*np.power(f[:,:,0]-2*f[:,:,1]+f[:,:,2],2) + 1/4*np.power(f[:,:,0]-4*f[:,:,1]+3*f[:,:,2],2)
        B2 = 13/12*np.power(f[:,:,1]-2*f[:,:,2]+f[:,:,3],2) + 1/4*np.power(f[:,:,1]-f[:,:,3],2)
        B3 = 13/12*np.power(f[:,:,2]-2*f[:,:,3]+f[:,:,4],2) + 1/4*np.power(3*f[:,:,2]-4*f[:,:,3]+f[:,:,4],2)
        #assign linear weights
        g1 = 1/10
        g2 = 3/5
        g3 = 3/10
        #compute the unscaled nonlinear weights
        wt1 = g1/np.power(ep+B1,2)
        wt2 = g2/np.power(ep+B2,2)
        wt3 = g3/np.power(ep+B3,2)
        wts = wt1 + wt2 + wt3
        #scale the nonlinear weights
        w1 = wt1/wts
        w2 = wt2/wts
        w3 = wt3/wts
        #compute the flux
        fl = f1*w1+f2*w2+f3*w3
        return fl
    FVM = FiniteVolumeMethodEuler(5, scheme)
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
            u_tmp = np.zeros_like(u[:,2])
            u_tmp[:] = u[:,2]
            for j in range(0,5):
                denominator = max_u - min_u
                safe_denominator = np.where(denominator == 0, 1.0, denominator)
                u[:,j] = np.where(denominator != 0, (u[:,j] - min_u) / safe_denominator, 0.0)

            fl[:,i] = model.predict(u).flatten()#compute \Delta u
            #fl = fl.flatten()
            fl[:,i] = np.multiply(fl[:,i],(max_u-min_u))+min_u
            fl[const_n,i] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethodEuler(5, scheme)
    return FVM
