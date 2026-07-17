# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:25:14 2019

@author: ben91
"""

from src.core.SimulationClasses import *
from src.core.TimeSteppingMethods import *
from src.core.FluxSplittingMethods import *
from src.core.Equations import *
from src.schemes import *
from src.initial_conditions.InitialConditions import *
from src.networks.wholeNetworks import *
from src.networks.LoadDataMethods import *
from src.viz.VisualizationFunctions import *

from tensorflow.keras import *
from tensorflow.keras.models import *
import numpy as np
import matplotlib.pyplot as plt
import math

K = 4
CFL = 1
P = 2
nx = 25*K
nt = int(25*P*K/CFL)+1
L = 2
T = 2*P
x = np.linspace(0,L,nx,endpoint=False)
t = np.linspace(0,T,nt)
dx = x[1]-x[0]
dt = t[1]-t[0]
train = 0
if(train):
    model = Const51stOrder(0.12)
    avgs = (loadInputData("2ndNewAvgs.csv"))
    flux = np.transpose(loadOutputData("2ndNewFlux.csv"))
    adm = optimizers.adam(lr=0.0001)
    model.compile(optimizer=adm,loss='mean_squared_error')
    model.fit(np.transpose(avgs),np.transpose(flux),epochs=10,batch_size=64,verbose = 1)
else:
    #model = load_model('tvm2.007427548332744swm20.h5')
    model = load_model('SeemsGood3.h5')

#model = load_model('BestRightNow.h5')

FVM1 = NNMethod(model)
FVM2 = ENO3()
EQ = adv()
FS = LaxFriedrichs(EQ, 1)
#FS = dontSplit(EQ)
IC = step1()
RK = SSPRK3()

testSim = Simulation(nx, nt, L, T, RK, FS, FVM1, IC)
WENO5Sim = Simulation(nx, nt, L, T, RK, FS, FVM2, IC)
uv = testSim.run()
uv_WENO5 = WENO5Sim.run()
'''
#specAnalysis(model, x, IC, RK, 'dense_402', 'add_74')
intError(1,x,t,uv,'Neural Network')
intError(1,x,t,uv_WENO5,'WENO5')

discTrackStep(1,x,t,uv_WENO5,T/2,'WENO5', -2-delt,-2+delt,True)
discTrackStep(1,x,t,uv,T/2,'Neural Network', -2-delt,-2+delt,True)

totalVariation(t,uv_WENO5,'WENO5')
totalVariation(t,uv,'Neural Network')

totalEnergy(t, uv_WENO5, dx, 'WENO5')
totalEnergy(t, uv, dx, 'Neural Network')
#make some visualizations
'''
plotDiscWidth(x,t,P,uv,uv_WENO5)

delt = 0.125



discTrackStep(1,x,t,uv_WENO5,T/2,'WENO5', -1-delt,-1+delt,True)
discTrackStep(1,x,t,uv,T/2,'Neural Network', -1-delt,-1+delt,True)

discTrackStep(1,x,t,uv_WENO5,T/2,'WENO5', -2-delt,-2+delt,True)
discTrackStep(1,x,t,uv,T/2,'Neural Network', -2-delt,-2+delt,True)

intError(1,x,t,uv,'Neural Network')
intError(1,x,t,uv_WENO5,'WENO5')

totalVariation(t,uv_WENO5,'WENO5')
totalVariation(t,uv,'Neural Network')

'''
totalVariation(t,uv_WENO5,'WENO5')
totalVariation(t,uv,'Neural Network')

totalEnergy(t, uv_WENO5, dx, 'WENO5')
totalEnergy(t, uv, dx, 'Neural Network')
plt.figure()
plt.plot(x,uv[:,4],'.')
plt.xlabel('x')
plt.ylabel('u')
'''
plt.plot(x,uv_WENO5[:,-1])


uv = testSim.run()
print("Simulation finished! Here is the final array:")
print(uv)

