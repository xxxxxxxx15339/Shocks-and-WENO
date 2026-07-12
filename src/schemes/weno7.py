# -*- coding: utf-8 -*- 
"""
Created on Satu 11 July 21:25:00 2026 

@author: Yasser BAOUZIL.
References
    ----------
    Dinshaw S Balsara, Chi-Wang Shu. Monotonicity Preserving Weighted Essentially Non-oscillatory Schemes
    with Increasingly High Order of Accuracy. Journal of Computational Physics, 2000, 160 (2), pp.405 - 452.
    ⟨10.1006/jcph.2000.6443⟩. ⟨hal-01634261⟩       
"""
from SimulationClasses import * 
import numpy as np 

def WENO7():
    def scheme(u):
        """
        vi+1/2 = v(xi+1/2) + O(Dx^2k-1)
        for k = 4: vi+1/2 = v(xi+1/2) + O(Dx^7) 7th order accuracy

        """
        ep = 1E-6 
    
        # Fluxes on sub stencils 
        f1 = -1/4*u[:,0] + 13/12*u[:,1] - 23/12*u[:,2] + 25/12*u[:,3]
        f2 = 1/12*u[:,1] - 5/12*u[:,2] + 13/12*u[:,3] + 1/4*u[:,4]
        f3 = -1/12*u[:,2] + 7/12*u[:,3] + 7/12*u[:,4] - 1/12*u[:,5]
        f4 = 1/4*u[:,3] + 13/12*u[:,4] - 5/12*u[:,5]+ 1/12*u[:,6]

        # Smoothness Inidicators: 
        B0 = (2107/240)*u[:,3]**2 - (1567/40)*u[:,3]*u[:,2] + (3521/120)*u[:,3]*u[:,1] - (309/40)*u[:,3]*u[:,0] + (11003/240)*u[:,2]**2 - (8623/120)*u[:,2]*u[:,1] + (2321/120)*u[:,2]*u[:,0] + (7043/240)*u[:,1]**2 - (647/40)*u[:,1]*u[:,0] + (547/240)*u[:,0]**2

        B1 = (3443/240)*u[:,3]**2 - (1261/120)*u[:,3]*u[:,4] - (2983/120)*u[:,3]*u[:,2] + (267/40)*u[:,3]*u[:,1] + (547/240)*u[:,4]**2 + (961/120)*u[:,4]*u[:,2] - (247/120)*u[:,4]*u[:,1] + (2843/240)*u[:,2]**2 - (821/120)*u[:,2]*u[:,1] + (89/80)*u[:,1]**2

        B2 = (3443/240)*u[:,3]**2 - (2983/120)*u[:,3]*u[:,4] + (267/40)*u[:,3]*u[:,5] - (1261/120)*u[:,3]*u[:,2] + (2843/240)*u[:,4]**2 - (821/120)*u[:,4]*u[:,5] + (961/120)*u[:,4]*u[:,2] + (89/80)*u[:,5]**2 - (247/120)*u[:,5]*u[:,2] + (547/240)*u[:,2]**2

        B3 = (2107/240)*u[:,3]**2 - (1567/40)*u[:,3]*u[:,4] + (3521/120)*u[:,3]*u[:,5] - (309/40)*u[:,3]*u[:,6] + (11003/240)*u[:,4]**2 - (8623/120)*u[:,4]*u[:,5] + (2321/120)*u[:,4]*u[:,6] + (7043/240)*u[:,5]**2 - (647/40)*u[:,5]*u[:,6] + (547/240)*u[:,6]**2

        # Linear Weights: 
        g1 = 1/35
        g2 = 12/35
        g3 = 18/35
        g4 = 4/35

        # Unscaled Nonlinear weights: 
        wt1 = g1/np.power(ep+B0, 2)
        wt2 = g2/np.power(ep+B1, 2)
        wt3 = g3/np.power(ep+B2, 2)
        wt4 = g4/np.power(ep+B3, 2)
        wts = wt1 + wt2 + wt3 + wt4

        # Scaled Nonlinear weights:
        w1 = wt1/wts
        w2 = wt2/wts
        w3 = wt3/wts 
        w4 = wt4/wts 

        # Compute the flux: 
        fl = f1*w1 + f2*w2 + f3*w3 + f4*w4 
        return fl 
    FVM = FiniteVolumeMethod(7, scheme)
    return FVM 


def NNMethod(model):    
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,3])
        u_tmp[:] = u[:,3]
        for i in range(0,7): 
            denominator = max_u - min_u
            safe_denominator = np.where(denominator == 0, 1.0, denominator)
            u[:,i] = np.where(denominator != 0, (u[:,i] - min_u) / safe_denominator, 0.0)

        fl = model.predict(u)#compute \Delta u
        fl = fl.flatten()
        fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethod(7, scheme)
    return FVM

def NNMethod_noScale(model):    
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,3])
        u_tmp[:] = u[:,3]
        #for i in range(0,5):
            #u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        fl = model.predict(u)#compute \Delta u
        fl = fl.flatten()
        #fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethod(7, scheme)
    return FVM


