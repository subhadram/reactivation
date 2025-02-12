#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:58:52 2023

@author: subhadramokashe
"""
import numpy as np
import matplotlib.pyplot as plt

N = 500
G_L = 10e-9;                
C = 100e-12;                
E_L = -70e-3;      
E_K = -80E-3         
V_Thresh = -50e-3;          
V_Reset = -80e-3;           
deltaT = 2e-3;              
tauw = 300e-3;   # Adaptation time constant (s)
taus = 20e-3           
a = 2e-9;                   # Adaptation recovery (S)
b = 0.02e-9;                # Adaptation strength (A)

I0 = 0e-9;                  # Baseline current
Iapp = 2e-9             # Applied current step

Vmax = 50e-3;               # Level of voltage to detect and crop spikes


dt = 1e-5           # dt in sec
tmax = 1              # maximum time in sec
tvector = np.arange(0,tmax,dt) 
I = np.random.normal(0,10e-8,[N,len(tvector)])
#I[200:400,:] = 0.0
#I[:,10000:] = 0.0
V= np.zeros((N,len(tvector)))
s= np.zeros((N,len(tvector)))
w =  np.zeros((N,len(tvector)))
spikes = np.zeros((N,len(tvector)))
gee = np.zeros((N,len(tvector)))
gei = np.zeros((N,len(tvector)))
gie = np.zeros((N,len(tvector)))
gii = np.zeros((N,len(tvector)))
V[0] = E_L


##network connectivity
J = np.zeros((N,N))
Jee1 = 140e-7
Jee2 = 280e-7
Jei = -35e-7
Jie = 20e-7
overlap = 0.1
group1 = 0.45
group2 = 0.45
inhgrp = 0.1
dw = 10e-7
ds = 10e-12
group1ei = int((group1)*N)
group2si = int((group1-overlap)*N)
group2ei = int((group1+group2- overlap)*N)
eind = np.zeros(N)
eind[0:group2ei] = int(1)
iind = np.zeros(N)
iind[group2ei:] = int(1)

J[:, :group2ei] = Jie
J[:,group2ei:] = Jei
J[:group1ei,:group1ei] = Jee1
J[group2si:group2ei,group2si:group2ei ] =  Jee2
p = 0.1
strc = np.array(np.where(np.random.rand(N, N) < p,1,0))  
J = np.multiply(J,strc)  
plt.imshow(J)
plt.show()
dgee = 20e-7
dgei = 20e-7
dgie = -20e-7
dgii = -20e-7
E_E = 0
E_I = -70e-3





for tt in range(len(tvector)-1):
    cellsfired = np.argwhere(V[:,tt] >= Vmax)
    cellsfirede = cellsfired[cellsfired< group2ei]
    cellsfiredi = cellsfired[cellsfired>= group2ei]
    #print(cellsfirede)
    V[cellsfired,tt] = V_Reset
    w[cellsfired,tt] = w[cellsfired,tt] + dw
    s[cellsfired,tt] = s[cellsfired,tt] + ds
    spikes[cellsfired,tt] =  1
    inputsfore = np.sum(J[cellsfirede,:],0)
    inputsfori = np.sum(J[cellsfiredi,:],0)
    gee[:,tt] = gee[:,tt] + dgee*inputsfore*eind
    gei[:,tt] = gei[:,tt] + dgei*inputsfore*iind
    gie[:,tt] = gie[:,tt] + dgie*inputsfori*eind
    gei[:,tt] = gii[:,tt] + dgii*inputsfori*iind
    V[:,tt + 1 ] = V[:,tt] + dt*( G_L*(E_L-V[:,tt] + deltaT*np.exp((V[:,tt]-V_Thresh)/deltaT) )+ w[:,tt]*(E_K-V[:,tt]) + I[:,tt]  + gee[:,tt]*(E_E - V[:,tt]) + gie[:,tt]*(E_E - V[:,tt])+ gei[:,tt]*(E_I - V[:,tt]) + gii[:,tt]*(E_I - V[:,tt]))/C 
    w[:,tt + 1] = w[:,tt]*np.exp(-dt/tauw)
    

plt.imshow(spikes[:,:2000],aspect = "auto", cmap = "gist_gray" )
plt.show()
import scipy
def spike_to_rate(spikes, window_std=100):
    window_size = np.arange(-3*window_std,3*window_std,1)
    window = scipy.stats.norm.pdf(window_size, 0, window_std)
    window /= window.sum()
    n_units = spikes.shape[0]
    estimate = np.zeros_like(spikes) # Create an empty array of the same size as spikes
    for i in range(n_units):
        y = np.convolve(window, spikes[i,:], mode='same')
        estimate[i,:] = y
    return estimate

rates = spike_to_rate(spikes, 100)


#plt.plot(tvector, 10**3*V[0,:])
plt.plot(tvector, np.sum(rates[0:225,:],0))
plt.plot(tvector, np.sum(rates[175:400,:],0))
plt.show()
