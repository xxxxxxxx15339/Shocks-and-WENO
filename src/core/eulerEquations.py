# -*- coding: utf-8 -*- 
"""
Created on Wed 8 July 03:41:25 2026 

@author: yasser ba 
Inspired by the work of the original author: ben91
"""

import numpy as np


# const_to_char and char_to_cons use the matrices to transform the standard physical variables (desity, momentum, energy) into a decoupled mathematical state called "characteristic variables", and back again
GAMMA = 1.4


def roe_eigenbasis(u, boundary='transmissive'):
    """Return one Roe-averaged left/right eigenbasis at every i+1/2 face."""
    if boundary == 'periodic':
        right_state = np.roll(u,-1,axis=0)
    elif boundary == 'transmissive':
        right_state = np.concatenate((u[1:,:], u[-1:,:]), axis=0)
    else:
        raise ValueError('Unsupported Euler boundary: {}'.format(boundary))
    rho_l, rho_r = u[:,0], right_state[:,0]
    vel_l, vel_r = u[:,1]/rho_l, right_state[:,1]/rho_r
    p_l = (GAMMA-1)*(u[:,2]-0.5*rho_l*vel_l**2)
    p_r = (GAMMA-1)*(right_state[:,2]-0.5*rho_r*vel_r**2)
    h_l = (u[:,2]+p_l)/rho_l
    h_r = (right_state[:,2]+p_r)/rho_r
    root_l, root_r = np.sqrt(rho_l), np.sqrt(rho_r)
    denominator = root_l+root_r
    velocity = (root_l*vel_l+root_r*vel_r)/denominator
    enthalpy = (root_l*h_l+root_r*h_r)/denominator
    sound_speed = np.sqrt((GAMMA-1)*(enthalpy-0.5*velocity**2))

    right = np.empty((len(u),3,3))
    right[:,0,:] = 1.0
    right[:,1,0] = velocity-sound_speed
    right[:,1,1] = velocity
    right[:,1,2] = velocity+sound_speed
    right[:,2,0] = enthalpy-velocity*sound_speed
    right[:,2,1] = 0.5*velocity**2
    right[:,2,2] = enthalpy+velocity*sound_speed
    left = np.linalg.inv(right)
    return left, right


def project_to_characteristic(values, left_eigenvectors):
    return np.einsum('nij,njk->nik', left_eigenvectors, values)


def project_to_conservative(values, right_eigenvectors):
    return np.einsum('nij,nj->ni', right_eigenvectors, values)

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
        alpha = np.max(np.abs(c)) #global Lax-Friedrichs splitting coefficient
        left_basis, right_basis = roe_eigenbasis(u, boundary=FVM.boundary)
        u_part = FVM.partU(u, offset=0)
        fu_part = FVM.partU(fu, offset=0)
        u_part_neg = np.flip(FVM.partU(u, offset=1), axis=2)
        fu_part_neg = np.flip(FVM.partU(fu, offset=1), axis=2)
            
        wp = project_to_characteristic(u_part, left_basis)
        fwp = project_to_characteristic(fu_part, left_basis)
        wm = project_to_characteristic(u_part_neg, left_basis)
        fwm = project_to_characteristic(fu_part_neg, left_basis)
        
        n,m,s = np.shape(wp)
        f_pos = np.zeros_like(fwp)
        f_neg = np.zeros_like(fwm)
        for i in range(0,m):
            f_pos[:,i,:] = 0.5*(fwp[:,i,:] + alpha*wp[:,i,:]) #positive half of flux
            f_neg[:,i,:] = 0.5*(fwm[:,i,:] - alpha*wm[:,i,:]) #negative half of flux
        f_half_pos = FVM.evalF(f_pos) #find the characteristic values at cell faces
        f_half_neg = FVM.evalF(f_neg) #find the characteristic values at cell faces
        flux_char = f_half_pos + f_half_neg
        flux_cons = project_to_conservative(flux_char, right_basis)

        if FVM.boundary == 'periodic':
            net_flux = flux_cons-np.roll(flux_cons,1,axis=0)
        else:
            net_flux = np.empty_like(flux_cons)
            net_flux[0,:] = flux_cons[0,:]-flux(u[0:1,:])[0,:]
            net_flux[1:,:] = flux_cons[1:,:]-flux_cons[:-1,:]
        return net_flux
    return full_flux
