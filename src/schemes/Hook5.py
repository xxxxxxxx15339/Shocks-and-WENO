# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:49:22 2019

@author: ben91
"""
# src/schemes/ENO3.py, Hook5.py, weno3.py, weno5.py, weno7.py
from ..core.SimulationClasses import FiniteVolumeMethod
import numpy as np

def Hook5():
    def scheme(u):
        fl = 1/30*u[:,0]-13/60*u[:,1]+47/60*u[:,2]+9/20*u[:,3]-1/20*u[:,4]
        return fl
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM
