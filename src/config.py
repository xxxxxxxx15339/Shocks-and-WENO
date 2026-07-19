"""Shared configuration for WENO-NN training and inference."""

# This repository currently ships five-point training data and a WENO5 model.
WENO_ORDER = 5

# A model trained with scaling must always be evaluated with scaling.
USE_SCALING = True

MODEL_PATH = 'trained_weno5.h5'
INPUT_DATA_PATH = 'data/2ndNewAvgs.csv'
OUTPUT_DATA_PATH = 'data/2ndNewFlux.csv'

RANDOM_SEED = 42
BOUNDARY_CONDITION = 'periodic'
MAX_CFL = 1.0
