#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:44:08 2023

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib
import pickle

taue = 10 #e membrane time constant
taui = 20 #i membrane time constant
vleake = -70 #e resting potential
vleaki = -62 #i resting potential
deltathe = 2 #eif slope parameter
C = 300 #capacitance
erev = 0 #e synapse reversal potential
irev = -75 #i synapse reversal potntial
vth0 = -52 #initial spike voltage threshold
ath = 10 #increase in threshold post spike
tauth = 30 #threshold decay timescale
vre = -75.0 #reset potential
taurefrac = 1 #absolute refractory period
aw_adapt = 10 #adaptation parameter a
bw_adapt = 2 #adaptation parameter b
tauw_adapt = 150 #adaptation timescale
dt = 0.1
tarray = np.arange(0,1000,dt)
gl = 10.0
N =1
V = np.zeros((len(tarray),N))
vth = np.zeros((len(tarray),N))
w = np.zeros((len(tarray),N))
I = 600.0
tr = np.zeros(N)
V[0] = -60+ np.random.randn(N)
w[0] = np.random.randn(N)
vth[0] = np.random.randn(N)
vpeak = 20.0

def sim():
    for t in range(len(tarray)-1):
        
        for i in range(0,N):
            
        
            dw = (aw_adapt*(V[t,i] - vleake) - w[t,i])*dt/tauw_adapt
           
            w[t+1,i] = w[t,i] + dw
        
            dvth = (vth0 - vth[t,i])*dt/tauth
        
            vth[t+1,i] = vth[t,i] + dvth
        
            if tr[i] > 0.0:
                V[t+1,:] = vre
                tr[i] = tr[i] -1
            elif V[t,i] >= vpeak:
                V[t+1,i] = vre
                tr[i] = taurefrac/dt
                w[t+1,i] = w[t,i] + bw_adapt
                vth[t+1,i] = vth[t,i] + ath
            
        
            else:
                dv = (-(V[t,i] - vleake) + deltathe*np.exp((V[t,i] - vth[t+1,i])/deltathe)) * (dt / taue) + (I - w[t+1,i])/C*dt
        
                V[t+1,i] = V[t,i] + dv
    return V
            
       

        
        
V = sim()        
#print(V)
#plt.plot(tarray, vth)
plt.plot(tarray, V[:,0])
plt.plot(tarray, V[:,0])
#plt.plot(tarray,w )
plt.show()