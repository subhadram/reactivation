#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:26:34 2024

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson

epsilon = np.arange(-1.0e-9, 1.1e-9, 2e-10)
trials = 100

with open('csfrac3s.pkl','rb') as f: 
    csfrac = np.squeeze(pickle.load(f,encoding = 'latin1'))
    
with open('isfrac3s.pkl','rb') as f: 
    isfrac = np.squeeze(pickle.load(f,encoding = 'latin1'))    


plt.errorbar(epsilon, np.mean(csfrac[:,:],1), yerr =np.std(csfrac[:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "forestgreen", label = "CS")
plt.errorbar(epsilon, np.mean(isfrac[:,:],1), yerr =np.std(isfrac[:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "gold", label = "IS")

plt.xlabel(r"$\epsilon$")
plt.ylabel("fraction of time in state")
plt.show()

# plt.errorbar(epsilon, np.mean(csrate[0,:,:],1), yerr =np.std(csrate[0,:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "forestgreen", label = "CS")
# plt.errorbar(epsilon, np.mean(israte[0,:,:],1), yerr =np.std(israte[0,:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "gold", label = "IS")

# plt.xlabel(r"$\epsilon$")
# plt.ylabel("mean firing rate")  
# plt.show()