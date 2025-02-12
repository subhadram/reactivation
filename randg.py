#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:07:43 2023

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt

N = 500
G_L = 5e-9
C = 100e-12
E_L = -70e-3
E_K = -80e-3
V_Thresh = -50e-3
V_Reset = -75e-3
deltaT = 2e-3
tauw = 300e-3   # Adaptation time constant (s)
taus = 20e-3
a = 2e-9                   # Adaptation recovery (S)
b = 0.02e-9                # Adaptation strength (A)

I0 = 0e-9                  # Baseline current
Iapp = 2e-9             # Applied current step

Vmax = 20e-3               # Level of voltage to detect and crop spikes


dt = 5e-5           # dt in sec
tmax = 20              # maximum time in sec
tvector = np.arange(0,tmax,dt)
I = np.sqrt(dt)*np.random.normal(0,25e-8,[N,len(tvector)])
#I[200:400,:] = 0.0
#I[:,10000:] = 0.0
V= np.zeros((N,len(tvector)))
s= np.zeros((N,len(tvector)))
w =  np.zeros((N,len(tvector)))
spikes = np.zeros((N,len(tvector)))

V[0] = E_L


J = np.zeros((N,N))
Jee1 = 2.5e-9
Jee2 = 1.5e-9
Jei = 5e-9
Jie = 2.5e-9
overlap = 0.1
group1 = 0.45
group2 = 0.45
dw = 3e-8
ds = 1
group1ind = int()
group1ei = int((group1)*N)
group2si = int((group1-overlap)*N)
group2ei = int((group1+group2- overlap)*N)
eind = np.zeros(N)
eind[0:group2ei] = int(1)
iind = np.zeros(N)
iind[group2ei:] = int(1)

J[group2ei:,:(group2ei-1)] = Jie
J[:,group2ei:] = Jei
J[:group1ei,:group1ei] = Jee1
J[group2si:group2ei,group2si:group2ei ] =  Jee2
p = 0.1
strc = np.array(np.where(np.random.rand(N, N) < p,1,0))
J = np.multiply(J,strc)
plt.imshow(J)
plt.show()

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
    inputsfrome = np.matmul(J[:,:group2ei-1],s[:group2ei-1,tt])
    inputsfromi = np.matmul(J[:,group2ei:],s[group2ei:,tt])
    V[:,tt + 1 ] = V[:,tt] + dt*( G_L*(E_L-V[:,tt] + deltaT*np.exp((V[:,tt]-V_Thresh)/deltaT) )+ w[:,tt]*(E_K-V[:,tt]) + I[:,tt]  + inputsfrome*(E_E - V[:,tt]) + inputsfromi*(E_I - V[:,tt]))/C
    s[:,tt+1] = s[:,tt]*np.exp(-dt/taus)
    w[:,tt+1] = w[:,tt]*np.exp(-dt/tauw)


plt.imshow(spikes[:,:],aspect = "auto", cmap = "gist_gray" )
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
    return estimate/(dt*window_std*6)

def spikes_to_rate(spikes, dt, window):
    windowsize = int(window/dt)
    sumspike = np.zeros_like(spikes)
    i = 0
    while i < len(spikes):
        sumspike[i:i+windowsize] = np.sum(spikes[i:i+windowsize])
        i = i + windowsize
        
    sumspike = sumspike/(window)
    return sumspike
        

rates = spike_to_rate(spikes, 100)
spikesg1 = np.mean(spikes[0:175,:],0)
spikesg2 = np.mean(spikes[225:400,:],0)
spikesov = np.mean(spikes[175:225,:],0)
rateg1 = spikes_to_rate(spikesg1, dt, 0.05)
rateg2 = spikes_to_rate(spikesg2, dt, 0.05)
rateov = spikes_to_rate(spikesov, dt, 0.05)

plt.plot(tvector, 10**3*V[100,:])
plt.plot(tvector, 10**3*V[400,:])
plt.show()

plt.plot(tvector, rateg1)
plt.plot(tvector, rateg2)
plt.plot(tvector, rateov)
plt.show()

plt.plot(tvector, np.mean(rates[0:174,:],0))
plt.plot(tvector, np.mean(rates[225:399,:],0))
plt.plot(tvector, np.mean(rates[175:224,:],0))
plt.show()

plt.plot(tvector, np.mean(rates[400:499,:],0))
plt.show()

plt.plot(tvector,s[1,:])
plt.plot(tvector,s[301,:])
plt.show()

plt.plot(inputsfrome)
plt.plot(inputsfromi)
plt.show()

