# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:24:10 2019

@author: ben91
"""

from .wholeNetworks import *
from keras import *
import numpy as np
import csv

def loadInputData(filename, stencil_size=5):
    with open(filename) as f:
        reader = csv.reader(f)
        #next(reader) # skip header
        avgs = [r for r in reader]
    avgs = np.asarray(avgs)
    avgs = avgs.astype(float)
    if stencil_size is not None:
        avgs = avgs[0:stencil_size,:]
    return avgs

def loadOutputData(filename):
    with open(filename) as g:
        reader2 = csv.reader(g)
        #next(reader) # skip header
        flux = [r for r in reader2]
    flux = np.asarray(flux)
    flux = flux.astype(float)
    return flux
