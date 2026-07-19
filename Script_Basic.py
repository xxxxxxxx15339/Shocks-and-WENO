# -*- coding: utf-8 -*-
"""Run the basic WENO-NN advection experiment."""

import argparse

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from src.config import BOUNDARY_CONDITION, MAX_CFL, MODEL_PATH, USE_SCALING
from src.core.Equations import adv
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation
from src.core.TimeSteppingMethods import SSPRK3
from src.initial_conditions.InitialConditions import step1
from src.schemes import NNMethod, NNMethod_noScale, WENO5
from src.viz.VisualizationFunctions import (
    discTrackStep,
    intError,
    plotDiscWidth,
    totalVariation,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the WENO5-NN comparison.')
    parser.add_argument('--model-path', default=MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
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

    model = load_model(args.model_path)

    FVM1 = NNMethod(model) if USE_SCALING else NNMethod_noScale(model)
    FVM2 = WENO5()
    if FVM1.boundary != BOUNDARY_CONDITION or FVM2.boundary != BOUNDARY_CONDITION:
        raise ValueError('Configured boundary does not match the selected methods.')
    EQ = adv()
    FS = LaxFriedrichs(EQ, 1)
    IC = step1()
    RK = SSPRK3()

    testSim = Simulation(nx, nt, L, T, RK, FS, FVM1, IC, max_cfl=MAX_CFL)
    WENO5Sim = Simulation(nx, nt, L, T, RK, FS, FVM2, IC, max_cfl=MAX_CFL)
    uv = testSim.run()
    uv_WENO5 = WENO5Sim.run()

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

    plt.plot(x,uv_WENO5[:,-1])

    print("Simulation finished! Here is the final array:")
    print(uv)


if __name__ == '__main__':
    main()
