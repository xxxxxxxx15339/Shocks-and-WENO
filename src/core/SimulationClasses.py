# -*- coding: utf-8 -*- 
"""
Created on Wed 8 July 03:41:25 2026 

@author: yasser ba 
Inspired by the work of the original author: ben91
"""

import numpy as np
import math
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, nx, nt, L, T, RK, FS, FVM, IC, max_cfl=None):
        self.nx = nx#number of gridpoints
        self.nt = nt#number of timesteps
        self.L = L#domain length
        self.T = T#time to simulate for
        self.RK = RK#timestepping method the sim will use
        self.FS = FS#flux splitting method the sim will use
        self.FVM = FVM#finite volume method the sim will use
        self.IC = IC#initial condition of the simulation
        self.max_cfl = max_cfl
        
    def run(self):
        x = np.linspace(0,self.L,self.nx,endpoint = False)
        t = np.linspace(0,self.T,self.nt,endpoint = True)
        dx = x[1]-x[0]
        dt = t[1]-t[0]
        if self.max_cfl is not None:
            speed = getattr(self.FS, 'max_wave_speed', None)
            if speed is None:
                raise ValueError('CFL validation requires a flux-splitting wave speed.')
            courant = abs(speed)*dt/dx
            if courant > self.max_cfl + 1e-12:
                raise ValueError(
                    'Unstable CFL number {} exceeds limit {}.'.format(
                        courant, self.max_cfl
                    )
                )
        u_all = np.zeros((int(self.nx),int(self.nt)))
        u_all[:,0] = self.IC(x)
        for i in range(0, int(self.nt-1)):
            u_all[:,i+1] = self.RK.stepIt(u_all[:,i], self.FS.flux, self.FVM, dt, dx)
        return u_all        
    
class eulerSimulation:
    def __init__(self, nx, nt, L, T, RK, flux, IC, neq,
                 enforce_positivity=True, max_wave_speed=None, max_cfl=None):
        self.nx = nx#number of gridpoints
        self.nt = nt#number of timesteps
        self.L = L#domain length
        self.T = T#time to simulate for
        self.RK = RK#timestepping method the sim will use
        self.flux = flux#finite volume method the sim will use
        self.IC = IC#initial condition of the simulation
        self.neq = neq#number of equations in system of pdes
        self.enforce_positivity = enforce_positivity
        self.max_wave_speed = max_wave_speed
        self.max_cfl = max_cfl

    def _validate_state(self, u, step):
        if not np.isfinite(u).all():
            raise FloatingPointError('Euler state became non-finite at step {}.'.format(step))
        if not self.enforce_positivity:
            return
        density = u[:,0]
        if np.any(density <= 0):
            raise FloatingPointError('Non-positive Euler density at step {}.'.format(step))
        velocity = u[:,1]/density
        pressure = (1.4-1)*(u[:,2]-0.5*density*np.power(velocity,2))
        if np.any(pressure <= 0):
            raise FloatingPointError('Non-positive Euler pressure at step {}.'.format(step))
        
    def runEuler(self):
        x = np.linspace(0,self.L,self.nx,endpoint = False)
        t = np.linspace(0,self.T,self.nt,endpoint = True)
        dx = x[1]-x[0]
        dt = t[1]-t[0]
        if self.max_cfl is not None:
            if self.max_wave_speed is None:
                raise ValueError('Euler CFL validation requires max_wave_speed.')
            courant = abs(self.max_wave_speed)*dt/dx
            if courant > self.max_cfl + 1e-12:
                raise ValueError(
                    'Unstable Euler CFL number {} exceeds limit {}.'.format(
                        courant, self.max_cfl
                    )
                )
        u_all = np.zeros((self.nx,self.nt,self.neq))
        u_all[:,0,:] = self.IC(x)
        self._validate_state(u_all[:,0,:], 0)
        for i in range(0, self.nt-1):#TODO: not a big deal but why nt-1?
            u_all[:,i+1,:] = self.RK.stepItEuler(u_all[:,i,:], self.flux, dt, dx, self.neq)
            self._validate_state(u_all[:,i+1,:], i+1)
        return u_all        
        
class TimeSteppingMethod:#assumes explicit method
    def __init__(self, ss, cff):
        self.ss = ss#substep coefficients
        self.cff = cff#coefficients of the timesteping method, should be lower triangular
        self.nss = ss.size#number of substeps the method has
        
    def stepIt(self, u, flux, FVM, dt, dx):#should work for a vector of inputs u
        n = u.size
        u_all = np.zeros((n,self.nss+1))
        u_all[:,0] = u
        for i in range(0,self.nss):
            for j in range(0,self.nss):
                u_all[:,i+1] += self.cff[i,j]*u_all[:,j]
            u_all[:,i+1] -= self.ss[i]*flux(u_all[:,i], FVM)*dt/dx#minus because HCL is du/dt = -df(u)/dx
        return u_all[:,-1]
    
    def stepItEuler(self, u, flux, dt, dx, neq):#FVM is now inside of the flux equation
        n = np.shape(u)[0]
        u_all = np.zeros((n,self.nss+1,neq))
        u_all[:,0,:] = u
        for i in range(0,self.nss):
            for j in range(0,self.nss):
                u_all[:,i+1,:] += self.cff[i,j]*u_all[:,j,:]
                #print(u_all[:,i+1,:])
            fl = flux(u_all[:,i,:])
            #print(fl)
            u_all[:,i+1,:] -= self.ss[i]*(fl)*dt/dx#minus because HCL is du/dt = -df(u)/dx
            #u_all[:,i+1,:] -= self.ss[i]*(fl)*dt/dx#minus because HCL is du/dt = -df(u)/dx
        return u_all[:,-1]
    
class FluxSplittingMethod:
    def __init__(self, Lp, Lm):
        self.Lp = Lp#positive flux. Note that this defines the PDE
        self.Lm = Lm#negative flux
        
    def flux(self, u, FVM):
        if getattr(FVM, 'boundary', 'periodic') != 'periodic':
            raise NotImplementedError(
                'Scalar flux splitting currently supports periodic boundaries only.'
            )
        u_int_og = FVM.evalF(u)#normal velocity
        u_int_fl = np.roll(np.flip(FVM.evalF(np.flip(u))),-1)#mirror velocity
        fp = self.Lp(u_int_og)
        fm = self.Lm(u_int_fl)
        
        '''
        plt.figure()
        plt.plot(u_int_og)
        plt.figure()
        plt.plot(u_int_fl)
        '''
        f = (fp+fm-np.roll(fp+fm,1))#flux in minus flux out
        return f
    
class FiniteVolumeMethod:
    def __init__(self, ss, L, boundary='periodic'):
        self.ss = ss#stencil size
        self.L = L#function that gives interpolated value
        if boundary != 'periodic':
            raise NotImplementedError('Only periodic scalar boundaries are implemented.')
        self.boundary = boundary
        
    def partU(self, u):#partition u into the stencil#TODO: get this going for systems of HCL (nx5x3 for euler eqn). pretend it works for now
        n = len(u)
        u_all = np.zeros((n,self.ss))
        
        for i in range(0,self.ss):#assume scheme is upwind or unbiased
            u_all[:,i] = np.roll(u,math.floor(self.ss/2)-i)
        return u_all
        
    def evalF(self, u):
        u_part = self.partU(u)
        u_int = self.L(u_part)
        #tv = np.sum(np.abs(u-np.roll(u, 1, axis = 0)),axis = 0)
        return u_int
    
class FiniteVolumeMethodEuler:
    def __init__(self, ss, L, boundary='transmissive'):
        self.ss = ss#stencil size
        self.L = L#function that gives interpolated value
        if boundary not in ('transmissive', 'periodic'):
            raise NotImplementedError(
                'Euler boundaries must be transmissive or periodic.'
            )
        self.boundary = boundary
        
    def partU(self, u, offset=0):
        """Build centered Euler stencils, filling ghost cells at boundaries."""
        cell_count, equation_count = np.shape(u)
        radius = math.floor(self.ss/2)
        if self.boundary == 'periodic':
            indices = (
                np.arange(cell_count)[:,None]
                + offset
                + np.arange(-radius, radius+1)[None,:]
            ) % cell_count
        else:
            indices = np.clip(
                np.arange(cell_count)[:,None]
                + offset
                + np.arange(-radius, radius+1)[None,:],
                0,
                cell_count-1,
            )
        gathered = u[indices,:]
        return np.transpose(gathered, (0,2,1))
        
    def evalF(self, f):
        '''
        inputs:
            f: characteristic fluxes (nx3x5)
        outputs:
            u_int: interpolated flux
        '''
        f_int = self.L(f)
        return f_int
        
