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
from tensorflow.keras import backend as be
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from sklearn.model_selection import train_test_split



print("Training started!", flush=True)

Kn = 4
CFL = 2/3
P = 3
nx = 25*Kn
nt = int(25*P*Kn/CFL)+1
L = 2
T = 2*P
x = np.linspace(0,L,nx,endpoint=False)
t = np.linspace(0,T,nt)
dx = x[1]-x[0]
dt = t[1]-t[0]
trainNetworks = True

avgs = np.transpose(loadInputData("data/2ndNewAvgs.csv"))
flux = loadOutputData("data/2ndNewFlux.csv")

X_train, X_val, y_train, y_val = train_test_split(
    avgs, flux, test_size=0.2, random_state=42, shuffle=True
)

EQ = adv()
FS = LaxFriedrichs(EQ, 1)
#FS = dontSplit(EQ)
IC = step1()
RK = SSPRK3()

#model = WENO51stOrder(0.3)
model = WENO51stOrder(0.15)
adm = optimizers.Adam(lr=0.001)
model.compile(optimizer=adm,loss='mean_squared_error')
early_stop = EarlyStopping(monitor='loss', patience=5, min_delta=1e-5, restore_best_weights=True)


L = x[-1] - x[0] + x[1] - x[0]
xg, tg = np.meshgrid(x,t)
xp = xg - tg
ons = np.ones_like(xp)
eex = np.greater(xp%L,ons)
while(trainNetworks):
    t1 = datetime.datetime.now()
    initial_weights = model.get_weights()
    for layr in range(12,20):
        if (layr%2)==0:
            f_in = np.shape(initial_weights[layr])[0]
            f_out = np.shape(initial_weights[layr])[1]
            limt = np.sqrt(6/(f_in+f_out))
            initial_weights[layr] = np.random.rand(f_in,f_out)*2*limt-limt
        else:
            f_in = np.shape(initial_weights[layr])[0]
            initial_weights[layr] = np.zeros(f_in)
    model.set_weights(initial_weights)    
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=150,
        callbacks=[early_stop],
        validation_data=(X_val, y_val),
        verbose=1
    )    

    #FVM1 = NNMethod(model)
    FVM1 = NNMethod_noScale(model)
    
    testSim = Simulation(nx, nt, L, T, RK, FS, FVM1, IC)
    uv = testSim.run()
    
    tvm,swm = evalPerf(x,t,P,uv,eex)
    if((tvm<2.016)and(swm<=20)):
        model.save('tvm'+str(tvm)+'swm'+str(swm)+'.h5')
        print('Saved')
        break
    t2 = datetime.datetime.now()
    print(t2-t1)
    #be.clear_session()
