# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:26:48 2019

@author: ben91
"""
import numpy as np
from keras import *
from keras import backend as K
from keras.layers import *


def _correction_model(stencil_size, hidden_size, classical_coefficients, regC):
    """Build the same conservative coefficient-correction NN used by WENO5."""
    inputs = Input(shape=(stencil_size,))
    coefficients = Lambda(classical_coefficients)(inputs)

    x1 = Dense(hidden_size, activation='relu')(coefficients)
    x2 = Dense(hidden_size, activation='relu')(x1)
    x3 = Dense(hidden_size, activation='relu')(x2)
    correction = Dense(
        stencil_size,
        activity_regularizer=regularizers.l2(regC),
    )(x3)
    corrected = Subtract()([coefficients, correction])

    # Add the same value to every coefficient so their sum remains one.
    consistency_kernel = np.full(
        (stencil_size, stencil_size),
        -1.0/stencil_size,
    )
    consistency_bias = np.full(stencil_size, 1.0/stencil_size)
    consistency = Dense(
        stencil_size,
        trainable=False,
        weights=[consistency_kernel, consistency_bias],
    )(corrected)
    all_coefficients = Add()([corrected, consistency])
    flux = dot([inputs, all_coefficients], axes=1, normalize=False)
    return Model(inputs=inputs, outputs=[flux])


def _weno3_coefficients(u):
    eps = 1e-6
    beta1 = K.square(u[:,1] - u[:,0])
    beta2 = K.square(u[:,2] - u[:,1])
    alpha1 = (1.0/3.0)/K.square(eps + beta1)
    alpha2 = (2.0/3.0)/K.square(eps + beta2)
    total = alpha1 + alpha2
    weights = K.stack([alpha1/total, alpha2/total], axis=1)
    candidates = K.constant([
        [-1.0/2.0, 3.0/2.0, 0.0],
        [0.0, 1.0/2.0, 1.0/2.0],
    ])
    return K.dot(weights, candidates)


def WENO31stOrder(regC):
    return _correction_model(3, 2, _weno3_coefficients, regC)


def _weno7_coefficients(u):
    eps = 1e-6
    b1 = (2107.0/240.0)*u[:,3]**2 - (1567.0/40.0)*u[:,3]*u[:,2] + (3521.0/120.0)*u[:,3]*u[:,1] - (309.0/40.0)*u[:,3]*u[:,0] \
         + (11003.0/240.0)*u[:,2]**2 - (8623.0/120.0)*u[:,2]*u[:,1] + (2321.0/120.0)*u[:,2]*u[:,0] \
         + (7043.0/240.0)*u[:,1]**2 - (647.0/40.0)*u[:,1]*u[:,0] + (547.0/240.0)*u[:,0]**2
    b2 = (3443.0/240.0)*u[:,3]**2 - (1261.0/120.0)*u[:,3]*u[:,4] - (2983.0/120.0)*u[:,3]*u[:,2] + (267.0/40.0)*u[:,3]*u[:,1] \
         + (547.0/240.0)*u[:,4]**2 + (961.0/120.0)*u[:,4]*u[:,2] - (247.0/120.0)*u[:,4]*u[:,1] \
         + (2843.0/240.0)*u[:,2]**2 - (821.0/120.0)*u[:,2]*u[:,1] + (89.0/80.0)*u[:,1]**2
    b3 = (3443.0/240.0)*u[:,3]**2 - (2983.0/120.0)*u[:,3]*u[:,4] + (267.0/40.0)*u[:,3]*u[:,5] - (1261.0/120.0)*u[:,3]*u[:,2] \
         + (2843.0/240.0)*u[:,4]**2 - (821.0/120.0)*u[:,4]*u[:,5] + (961.0/120.0)*u[:,4]*u[:,2] \
         + (89.0/80.0)*u[:,5]**2 - (247.0/120.0)*u[:,5]*u[:,2] + (547.0/240.0)*u[:,2]**2
    b4 = (2107.0/240.0)*u[:,3]**2 - (1567.0/40.0)*u[:,3]*u[:,4] + (3521.0/120.0)*u[:,3]*u[:,5] - (309.0/40.0)*u[:,3]*u[:,6] \
         + (11003.0/240.0)*u[:,4]**2 - (8623.0/120.0)*u[:,4]*u[:,5] + (2321.0/120.0)*u[:,4]*u[:,6] \
         + (7043.0/240.0)*u[:,5]**2 - (647.0/40.0)*u[:,5]*u[:,6] + (547.0/240.0)*u[:,6]**2

    linear_weights = (1.0/35.0, 12.0/35.0, 18.0/35.0, 4.0/35.0)
    betas = (b1, b2, b3, b4)
    alphas = [weight/K.square(eps + beta) for weight, beta in zip(linear_weights, betas)]
    total = alphas[0] + alphas[1] + alphas[2] + alphas[3]
    weights = K.stack([alpha/total for alpha in alphas], axis=1)
    candidates = K.constant([
        [-1.0/4.0, 13.0/12.0, -23.0/12.0, 25.0/12.0, 0.0, 0.0, 0.0],
        [0.0, 1.0/12.0, -5.0/12.0, 13.0/12.0, 1.0/4.0, 0.0, 0.0],
        [0.0, 0.0, -1.0/12.0, 7.0/12.0, 7.0/12.0, -1.0/12.0, 0.0],
        [0.0, 0.0, 0.0, 1.0/4.0, 13.0/12.0, -5.0/12.0, 1.0/12.0],
    ])
    return K.dot(weights, candidates)


def WENO71stOrder(regC):
    return _correction_model(7, 4, _weno7_coefficients, regC)


def WENO51stOrder(regC):
    pntsuse = 5
    wub6 = np.array([[ 1, 1, 0, 0, 0, 0],
                     [-2,-4, 1, 1, 0, 0],
                     [ 1, 3,-2, 0, 1, 3],
                     [ 0, 0, 1,-1,-2,-4],
                     [ 0, 0, 0, 0, 1, 1]])
    wub6c = np.array([ 0, 0, 0, 0, 0, 0])
    eps = 1e-6
    wub3 = np.array([[13/12,0,0],
                     [1/4  ,0    ,0],
                     [0    ,13/12,0],
                     [0    ,1/4  ,0],
                     [0    ,0    ,13/12],
                     [0    ,0    ,1/4]])
    wub3c = np.array([eps, eps, eps])
    wba1 = np.array([[0.1],
                     [0],
                     [0]])
    wba1c = np.array([0])
    wba2 = np.array([[0],
                     [0.6],
                     [0]])
    wba2c = np.array([0])
    wba3 = np.array([[0],
                     [0],
                     [0.3]])
    wba3c = np.array([0])
    w_to_c = np.array([[1/3, -7/6, 11/6, 0,0],
                     [0, -1/6, 5/6, 1/3, 0],
                     [0, 0, 1/3, 5/6, -1/6]])
    w_to_cc = np.array([0,0,0,0,0])
    wub1 = np.array([[-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5]])
    wub1c = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
        
    # Make weights for the projection
    u_05 = Input(shape = (5, ))#merge all the average inputs as u_(-2),u_(-1),u_0,u_1,u_2,u_3
        
    b_6 = Dense(6,trainable=False,weights=[wub6,wub6c])(u_05)#6 differences for smoothness indicators
    b_6s = Lambda(lambda x: (x ** 2))(b_6)#6 differences squared
    
    b_3 = Dense(3,trainable=False,weights=[wub3,wub3c])(b_6s)#3 smoothness indicators + epsilon
    b_3i = Lambda(lambda x: 1/(x ** 2))(b_3)#invert and square the smoothness indicators
    
    a_1 = Dense(1,trainable=False,weights=[wba1,wba1c])(b_3i)#1st unscaled nonlinear weight
    a_2 = Dense(1,trainable=False,weights=[wba2,wba2c])(b_3i)#2nd unscaled nonlinear weight
    a_3 = Dense(1,trainable=False,weights=[wba3,wba3c])(b_3i)#3rd unscaled nonlinear weight
    
    a_s = Add()([a_1,a_2,a_3])#sum of nonlinear weights
    a_si = Lambda(lambda x: 1/(x))(a_s)#invert the sum of weights
    
    w1 = Multiply()([a_1,a_si])#scale the 1st smoothness indicator
    w2 = Multiply()([a_2,a_si])#scale the 2nd smoothness indicator
    w3 = Multiply()([a_3,a_si])#scale the 3rd smoothness indicator
    
    w = concatenate([w1,w2,w3])
    
    Cs = Dense(5,trainable=False,weights=[w_to_c,w_to_cc])(w)#Final WENO5 coefficients
    
    x1 = Dense(3,activation='relu')(Cs)
    x2 = Dense(3,activation='relu')(x1)
    x3 = Dense(3,activation='relu')(x2)
    #TODO: Pass arguments to this function that define the regularization and neural network nodes/layers and l1/l2 optimization
    dc = Dense(5,activity_regularizer=regularizers.l2(regC))(x3)#end the DNN, the 5 differences are the outputs
    c_tilde = Subtract()([Cs,dc])#use the differences to modify the coefficients
    
    dc2 = Dense(pntsuse,trainable=False,weights=[wub1,wub1c])(c_tilde)#compute how each coefficient must be changed for consistency
    
    c_all = Add()([c_tilde,dc2])
    
    p2 = dot([u_05,c_all], axes = 1, normalize = False)#compute flux from all 5 coefficients
    
    model = Model(inputs=u_05, outputs=[p2])
    return model

def Const51stOrder(regC):
    pntsuse = 5

    H51 = np.array([[0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0]])
    H51c = np.array([1/30, -13/60, 47/60, 9/20, -1/20])
        
    wub1 = np.array([[-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5]])
    wub1c = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    
    
    # Make weights for the projection
    u_05 = Input(shape = (5, ))#merge all the average inputs as u_(-2),u_(-1),u_0,u_1,u_2,u_3
        
    Cs = Dense(5,trainable=False,weights=[H51,H51c])(u_05)#Final WENO5 coefficients
    reggersA = 0.001
    reggersb = 0.001
    
    x1 = Dense(5,activation='relu')(u_05)
    x2 = Dense(5,activation='relu')(x1)
    x3 = Dense(5,activation='relu')(x2)

    #TODO: Pass arguments to this function that define the regularization and neural network nodes/layers and l1/l2 optimization
    #dc = Dense(5,activity_regularizer=regularizers.l2(regC))(x9)#end the DNN, the 5 differences are the outputs
    dc = Dense(5,trainable=False,activity_regularizer=regularizers.l2(regC))(x3)#end the DNN, the 5 differences are the outputs
    c_tilde = Subtract()([Cs,dc])#use the differences to modify the coefficients
    
    dc2 = Dense(pntsuse,trainable=False,weights=[wub1,wub1c])(c_tilde)#compute how each coefficient must be changed for consistency
    
    c_all = Add()([c_tilde,dc2])
    
    p2 = dot([u_05,c_all], axes = 1, normalize = False)#compute flux from all 5 coefficients
    #p2 = dot([u_05,Cs], axes = 1, normalize = False)#compute flux from all 5 coefficients
    
    model = Model(inputs=u_05, outputs=[p2])
    return model
