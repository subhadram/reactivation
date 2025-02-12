#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:42:48 2024

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib
import pickle


with open('cstrace1_1.pkl','rb') as f: 
    cstrace1 = np.squeeze(pickle.load(f,encoding = 'latin1'))
    
with open('cstrace2_1.pkl','rb') as f: 
    cstrace2 = np.squeeze(pickle.load(f,encoding = 'latin1'))
    
with open('istrace1_1.pkl','rb') as f: 
    istrace1 = np.squeeze(pickle.load(f,encoding = 'latin1'))
    
with open('istrace2_1.pkl','rb') as f: 
    istrace2 = np.squeeze(pickle.load(f,encoding = 'latin1'))
    
    
with open('intrace1_1.pkl','rb') as f: 
    intrace1 = np.squeeze(pickle.load(f,encoding = 'latin1'))
    
with open('intrace2_1.pkl','rb') as f: 
    intrace2 = np.squeeze(pickle.load(f,encoding = 'latin1'))
    
dt = 1e-4           # dt in sec
tmax = 100          # maximum time in sec
tvector = np.arange(0,tmax,dt)
    
ax1 = plt.subplot(311)    
plt.plot(tvector,cstrace1,color = "forestgreen")
plt.plot(tvector,cstrace2,color = "yellowgreen")
ax2 = plt.subplot(312,sharex = ax1)
plt.plot(tvector,istrace1,color = "gold")
plt.plot(tvector,istrace2,color = "khaki")
ax3 = plt.subplot(313,sharex = ax1)
plt.plot(tvector,intrace1, color = "navy")
plt.plot(tvector,intrace2, color = "slateblue")
plt.show()