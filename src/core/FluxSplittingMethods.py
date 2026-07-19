# -*- coding: utf-8 -*- 
"""
Created on Thur 9 July 13:18:00 2026 

@author: yasser ba 
Inspired by the work of the original author: ben91
"""


from .SimulationClasses import FluxSplittingMethod
import numpy as np

def LaxFriedrichs(F,alpha):
    def Lp(u):
        return 0.5*(F(u) + alpha*u)
    def Lm(u):
        return 0.5*(F(u) - alpha*u)
    FS = FluxSplittingMethod(Lp, Lm)
    FS.max_wave_speed = abs(alpha)
    return FS

def dontSplit(F):
    def Lp(u):
        return F(u)
    def Lm(u):
        return 0
    FS = FluxSplittingMethod(Lp, Lm)
    return FS
