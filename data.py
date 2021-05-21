#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:16:48 2021

@author: raphaelbailly
"""

import numpy as np

data = {}
data['input'] = np.array([[ 2.51556767e+00,  1.83353912e+00,  1.75147133e+00],
       [ 2.83222365e+00,  1.29824660e+00,  8.11566111e-01],
       [ 2.65818596e+00,  2.57889358e+00,  1.12616895e+00],
       [ 3.11955467e+00,  1.67334447e+00, -4.10185037e-03],
       [ 2.15949526e+00,  2.66865532e+00,  2.47579681e-01],
       [ 3.00050280e+00,  1.01834429e+00,  1.42587047e+00],
       [ 3.18646209e+00,  2.30273411e+00,  5.90942832e-01],
       [ 3.06511426e+00,  1.08275023e+00,  6.61518438e-01],
       [ 3.06695878e+00,  1.40746852e+00,  9.32713049e-01],
       [ 2.20798248e+00,  2.01881720e+00,  1.10318103e+00],
       [ 3.15817538e+00,  1.48983482e+00,  8.22792648e-01],
       [ 2.99095229e+00,  3.46804580e-01,  6.45382484e-01],
       [ 1.62528998e+00,  3.11955543e+00,  8.51280403e-01],
       [ 2.67387806e+00,  2.93901211e+00,  1.99689713e+00],
       [ 2.62335423e+00,  2.77220362e+00,  1.32961270e+00],
       [ 3.36608413e+00,  3.11700012e+00,  1.86064365e+00],
       [ 2.73722011e+00,  2.92844891e+00,  3.91863038e-01],
       [ 2.97689509e+00,  2.42595635e+00,  1.73903529e+00],
       [ 3.34008931e+00,  1.20653431e+00,  8.38343152e-01],
       [ 3.28674504e+00,  1.28053888e+00,  1.56573721e+00],
       [ 2.21385437e+00,  6.31706784e-01,  3.76716283e+00],
       [ 1.62713029e+00,  7.03429947e-01,  2.68977323e+00],
       [ 9.45169981e-01,  3.90973011e-01,  2.30967126e+00],
       [ 1.01135876e+00,  1.61445565e+00,  3.35166427e+00],
       [ 4.89231500e+00,  1.25035250e+00,  2.15379105e+00],
       [ 1.49994297e+00,  1.02612260e+00,  3.28591576e+00],
       [ 8.19651127e-01,  1.00949070e+00,  1.85835336e+00],
       [ 2.41027885e+00,  5.95637742e-01,  2.60076543e+00],
       [ 1.82820639e+00,  4.15971425e-01,  2.09484166e+00],
       [ 3.14566148e+00,  1.37733014e+00,  2.44478547e+00],
       [ 3.22125149e+00,  2.04383266e+00,  2.52038302e+00],
       [ 1.68275993e+00,  2.35184871e-01,  3.79080394e+00],
       [ 1.72754081e+00,  1.57675035e+00,  2.22670022e+00],
       [ 2.11001984e+00,  1.03344061e+00,  2.59778582e+00],
       [ 1.08654153e+00,  8.33753697e-01,  2.29274762e+00],
       [-4.90137582e-01,  1.10404191e+00,  2.92791906e+00],
       [ 2.14539671e+00,  6.73958993e-01,  3.50184896e+00],
       [ 9.79102628e-01,  1.87918344e+00,  3.25028250e+00],
       [ 1.64233754e+00,  1.21367628e+00,  2.11701285e+00],
       [ 1.02192235e+00,  4.38126963e-01,  2.50221572e+00],
       [ 2.63155917e+00,  7.03167193e-01,  2.47605018e+00],
       [ 1.06170434e+00,  1.06228586e+00,  2.70973764e+00],
       [ 2.01832369e+00,  2.99567404e-01,  3.21995629e+00],
       [ 3.24254050e+00,  7.89086699e-01,  2.45491410e+00],
       [ 2.27245144e+00,  1.21731780e+00,  3.27586910e+00],
       [ 2.25871391e+00,  2.01621758e+00,  3.10881670e+00],
       [ 2.78440458e+00,  8.85204513e-01,  3.07043952e+00],
       [-1.42671317e-01,  4.01181405e-01,  3.29247313e+00],
       [ 8.06465324e-01,  7.99360234e-02,  3.21864362e+00],
       [ 1.89050090e-01,  9.27698899e-01,  2.72442538e+00],
       [ 1.20527954e+00,  3.52952622e+00,  7.10666741e-01],
       [ 2.33873119e+00,  3.06166667e+00,  3.33795884e+00],
       [ 2.36275547e+00,  3.00072334e+00,  2.57357214e+00],
       [ 5.40561592e-01,  3.44747183e+00,  1.29440797e+00],
       [ 1.17664295e+00,  3.62066035e+00,  2.55085164e+00],
       [ 7.84181075e-01,  2.33019092e+00,  1.47195645e+00],
       [ 1.93414495e+00,  2.99371542e+00,  1.49136959e+00],
       [ 1.12533415e+00,  2.79050248e+00,  1.99315854e+00],
       [ 1.60779181e+00,  3.56916012e+00,  1.61856480e+00],
       [ 1.53650400e+00,  2.87677935e+00,  2.21113618e+00],
       [ 1.56524336e+00,  2.49568096e+00,  2.55695762e+00],
       [ 1.24605510e+00,  2.43603851e+00,  2.99982564e+00],
       [ 6.32579083e-01,  3.48207086e+00,  1.33622487e+00],
       [ 1.01146364e+00,  2.62820906e+00,  2.64266274e+00],
       [ 1.74687189e+00,  3.69684059e+00,  1.39731396e+00],
       [-2.65757474e-01,  3.25624781e+00,  1.38487355e+00],
       [ 1.32935899e+00,  2.42143189e+00,  1.93644724e+00],
       [ 4.47994057e-01,  2.85616547e+00,  3.47089271e+00],
       [ 1.45484429e+00,  2.55574683e+00,  4.01533067e+00],
       [ 9.26279540e-01,  3.15407091e+00,  1.03049290e-02],
       [ 6.77770126e-01,  3.69890652e+00,  1.70526448e+00],
       [ 8.22061183e-01,  3.62802915e+00,  3.23478380e+00],
       [ 2.42694275e-01,  2.74894115e+00,  8.62036406e-01],
       [ 1.02326494e+00,  2.07752799e+00,  2.55491778e+00],
       [ 3.44579209e-01,  2.57037869e+00,  2.72982530e+00],
       [ 2.69188027e+00,  2.03773840e+00,  3.90312537e+00],
       [ 3.89040765e+00,  2.47358427e+00,  4.47172016e+00],
       [ 2.77484793e+00,  1.77021622e+00,  2.22860598e+00],
       [ 3.19268900e+00,  3.57216648e+00,  2.57966238e+00],
       [ 3.66362888e+00,  3.51838698e+00,  3.50695630e+00],
       [ 2.95496767e+00,  2.62691952e+00,  3.53508314e+00],
       [ 3.27301889e+00,  3.14684421e+00,  2.72145940e+00],
       [ 4.03420344e+00,  3.23521983e+00,  3.56043646e+00],
       [ 2.83640180e+00,  3.34796736e+00,  2.70831306e+00],
       [ 3.35026324e+00,  3.02461328e+00,  2.75656903e+00],
       [ 2.98790540e+00,  3.57890598e+00,  2.95862674e+00],
       [ 4.08368219e+00,  3.24052280e+00,  2.77130810e+00],
       [ 3.36022053e+00,  2.37123742e+00,  2.02846699e+00],
       [ 2.81445119e+00,  2.16869307e+00,  3.18900229e+00],
       [ 3.14211826e+00,  3.04433457e+00,  3.91719520e+00]])

data['target'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
