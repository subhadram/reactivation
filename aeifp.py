#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 18:18:18 2023

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
dt  = dt = 1e-6


G_L = 10e-9               # Leak conductance (S)
C = 100e-12               # Capacitance (F) 
E_L = -70e-3             # Leak potential (V)
V_Thresh = -50e-3         # Threshold potential (V)
V_Reset = -80e-3           # Reset potential (V)
deltaT = 2e-3             # Threshold shift factor (V)
tauw = 200e-3              # Adaptation time constant (s)
a = 2e-9                   # Adaptation recovery (S)
b = 0.02e-9            # Adaptation strength (A)

I0 = 0e-9                 #Baseline current
Iapp = 0.221e-8             #Applied current step

Vmax = 50e-3              # Level of voltage to detect and crop spikes

#Simulation set-up

dt = 1e-6                 # dt in sec
tmax = 3                  # maximum time in sec
tarray = np.arange(0,10,dt)       # vector of time points

ton = 0.5                  # time to add step current
toff = 2.5                # time to remove step current
non = int(ton/dt)     # time-point corresponding to ton
noff = int(toff/dt)      # time-point corresponding to toff
I = I0*np.ones(len(tarray)) # applied current initialized to baseline
I[non:noff] = Iapp        # add the step to the applied current vector    
v = np.zeros(len(tarray))      # initialize membrane potential at all time-points
v[0] = E_L                   # set initial value to be the leak potential
w = np.zeros(len(tarray))       # initialize adaptation variable at all time-points
spikes = np.zeros(len(tarray)) 

for i in range(0,len(tarray)-1):
    if v[i] > Vmax:
        v[i] = V_Reset
        w[i] = w[i] + b
        spikes[i] = 1
    v[i+1] = v[i] + dt*( G_L*(E_L-v[i] + deltaT*np.exp((v[i]-V_Thresh)/deltaT) )- w[i] + I[i])/C
    w[i+1] = w[i] +  dt*( a*(v[i]-E_L) - w[i] )/tauw
    
    
plt.plot(tarray, 10**3*v)
plt.show()