# -*- coding: utf-8 -*- 
"""
Created on Wed 8 July 03:41:25 2026 

@author: yasser ba 
Inspired by the work of the original author: ben91
"""

import numpy as np


# const_to_char and char_to_cons use the matrices to transform the standard physical variables (desity, momentum, energy) into a decoupled mathematical state called "characteristic variables", and back again
def cons_to_char(u,f):
    '''
    transform the conservative variables to the characteristic variables
    inputs:
        u: vector of conservative variables (n x 3 matrix)
        f: vector of conservative fluxes (n x 3 matrix)
    outputs:
        w: vector of characteristic variables (n x 3 matrix)
    '''
    n,m,s = np.shape(u)
    g = 1.4#ratio of specific heats for air
    us = 0.5*(u[:,:,int(np.floor(s/2))] + u[:,:,int(np.ceil(s/2))])#average to find the state
    rho = us[:,0]#density
    vel = us[:,1]/us[:,0]#velocity
    E = us[:,2]#energy
    p = (g-1)*(E-0.5*rho*np.power(vel,2))#pressure
    c = np.sqrt(g*p/rho)#sound speed
    beta = 1/(np.sqrt(2)*rho*c)
    L = np.zeros((n,m,m))#vector of 3x3 transformation matrices
    
    L[:,0,0] = 1-(g-1)/2*np.power(vel/c,2)
    L[:,0,1] = (g-1)*vel/np.power(c,2)#TODO: change syntax to python
    L[:,0,2] = -(g-1)/np.power(c,2)
    L[:,1,0] = beta*(0.5*(g-1)*np.power(vel,2)-c*vel)
    L[:,1,1] = beta*(c-(g-1)*vel)
    L[:,1,2] =  beta*(g-1)
    L[:,2,0] = beta*(0.5*(g-1)*np.power(vel,2)+c*vel)
    L[:,2,1] = -beta*(c+(g-1)*vel)
    L[:,2,2] = beta*(g-1)
    
    w = np.zeros_like(u)
    fw = np.zeros_like(f)
    for i in range(0,n):
        w[i,:,:] = np.matmul(L[i,:,:],u[i,:,:])#TODO: check if we need  any sort of transposes here
        fw[i,:,:] = np.matmul(L[i,:,:],f[i,:,:])
    return w,fw


def char_to_cons(fc,u,boundary='transmissive'):
    '''
    transform the conservative variables to the characteristic variables
    inputs:
        fc: characteristic fluxes (n x 3 matrix)
        u: vector of conserved variables (n x 3 matrix)
    outputs:
        f_cons: vector of conserved fluxes (n x 3 matrix)
    '''
    if boundary == 'periodic':
        right_state = np.roll(u,-1,axis=0)
    elif boundary == 'transmissive':
        right_state = np.concatenate((u[1:,:], u[-1:,:]), axis=0)
    else:
        raise ValueError('Unsupported Euler boundary: {}'.format(boundary))
    us = 0.5*(u + right_state)
    g = 1.4 #ratio of specific heats for air
    rho = us[:,0] #density
    vel = us[:,1]/us[:,0] #velocity
    E = us[:,2] #energy
    p = (g-1)*(E-0.5*rho*np.power(vel,2)) #pressure
    c = np.sqrt(g*p/rho) #sound speed
    alpha = rho/(np.sqrt(2)*c)
    n,m = np.shape(u)
    R = np.zeros((n,m,m)) #vector of 3x3 transformation matrices
    R[:,0,0] = 1
    R[:,0,1] = alpha
    R[:,0,2] = alpha
    R[:,1,0] = vel
    R[:,1,1] = alpha*(vel+c)
    R[:,1,2] = alpha*(vel-c)
    R[:,2,0] = 0.5*np.power(vel,2)
    R[:,2,1] = alpha*(0.5*np.power(vel,2)+np.power(c,2)/(g-1)+c*vel)
    R[:,2,2] = alpha*(0.5*np.power(vel,2)+np.power(c,2)/(g-1)-c*vel)
    f_cons = np.zeros_like(fc)
    for i in range(0,n):
        f_cons[i,:] = np.matmul(R[i,:,:],fc[i,:])
        
    return f_cons

def flux(u):
    g =  1.4
    u1 = u[:,0]
    u2 = u[:,1]
    u3 = u[:,2]
    f = np.zeros_like(u)        
    f[:,0] = u2
    f[:,1] = 0.5*(3-g)*np.power(u2,2)/u1+(g-1)*u3
    f[:,2] = g*u2*u3/u1-0.5*(g-1)*np.power(u2,3)/np.power(u1,2)
    return f

def spds(ws):
    g = 1.4 #ratio of specific heats for air
    rho = ws[:,0] #density
    vel = ws[:,1]/ws[:,0] #velocity
    E = ws[:,2] #energy
    p = (g-1)*(E-0.5*rho*np.power(vel,2)) #pressure
    c = np.sqrt(g*p/rho) #sound speed
    sps = np.zeros_like(ws)
    sps[:,0] = vel
    sps[:,1] = vel + c
    sps[:,2] = vel - c
    return sps

def getEulerFlux(FVM): #this contains the flux splitting in it so no nead to do more flux splitting
    def full_flux(u):
        '''
        inputs:
            u: cell average conservative variables
        outputs:
            flux: the flux at the boundary
        '''
        fu = flux(u) #compute flux from conserved variables cell averages (the temporary one)
        c = spds(u) #compute wave speeds from cell averages
        alph = np.max(np.abs(c),axis=0) #flux splitting coefficient
        u_part = FVM.partU(u, offset=0)
        fu_part = FVM.partU(fu, offset=0)
        u_part_neg = np.flip(FVM.partU(u, offset=1), axis=2)
        fu_part_neg = np.flip(FVM.partU(fu, offset=1), axis=2)
            
        wp,fwp = cons_to_char(u_part,fu_part) #project to characteristic variables
        wm,fwm = cons_to_char(u_part_neg,fu_part_neg)
        
        n,m,s = np.shape(wp)
        f_pos = np.zeros_like(fwp)
        f_neg = np.zeros_like(fwm)
        for i in range(0,m):
            f_pos[:,i,:] = 0.5*(fwp[:,i,:] + alph[i]*wp[:,i,:]) #positive half of flux
            f_neg[:,i,:] = 0.5*(fwm[:,i,:] - alph[i]*wm[:,i,:]) #negative half of flux
        f_half_pos = FVM.evalF(f_pos) #find the characteristic values at cell faces
        f_half_neg = FVM.evalF(f_neg) #find the characteristic values at cell faces
        flux_char = f_half_pos + f_half_neg
        flux_cons = char_to_cons(flux_char,u,boundary=FVM.boundary)

        if FVM.boundary == 'periodic':
            net_flux = flux_cons-np.roll(flux_cons,1,axis=0)
        else:
            net_flux = np.empty_like(flux_cons)
            net_flux[0,:] = flux_cons[0,:]-flux(u[0:1,:])[0,:]
            net_flux[1:,:] = flux_cons[1:,:]-flux_cons[:-1,:]
        return net_flux
    return full_flux
