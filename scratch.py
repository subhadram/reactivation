#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:43:07 2023

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt


dt = 0.1
tarray = np.arange(0,1000,dt)
N = 10
v = np.zeros((N,len(tarray)))
w = np.zeros((N,len(tarray)))
vth = np.zeros((N,len(tarray)))
lastspiked = 0
taurefrac = 3.0
aw_adapt = 4 #adaptation parameter a
bw_adapt = 0.0805#adaptation parameter b
tauw_adapt = 150.0#adaptation timescale
vleake = -70
vth0 = -52
tauth = 30
taue = 20
deltathe = 2
C = 300
IT = 600
vpeak = 20
ath = 10
vre = -60
v[0,0] = -60
I = np.zeros(len(tarray))
I[:] = IT
spikeda = []
jex = 1.78
rx = 4.0
jex = 1.78
tauerise = 1.0
tauedecay = 6.0
erev = 0
w[0,0] = 10
nextexspike = np.zeros(N)
lastspiked =  np.zeros(N)
inpre = np.zeros(N)
xerise = np.zeros(N)
xedecay = np.zeros(N)
for t in range(len(tarray)-1):
    
    tt = t*dt
    #print(max(v[:,t]))
    
    for i in range(0,N):
        dw = (aw_adapt*(v[i,t] - vleake) - w[i,t])*dt/tauw_adapt
        w[i,t+1] = w[i,t] + dw
        
        dvth = (vth0 - vth[i,t])*dt/tauth
        
        vth[i,t+1] = vth[i,t] + dvth
        while tt > nextexspike[i]:
            nextexspike[i] +=  -np.log(1-np.random.rand()/rx)
            inpre[i] +=  jex
        xerise[i] += -dt*xerise[i]/tauerise + inpre[i]
        xedecay[i] += -dt*xedecay[i]/tauedecay + inpre[i]
        if (tt > lastspiked[i]+ taurefrac):
            ge = (xedecay[i] - xerise[i])/(tauedecay - tauerise)
            dv = (vleake - v[i,t])/taue + deltathe*np.exp((v[i,t]-vth0)/deltathe)/taue +  w[i,t]/C#  ge*(erev-v[i,t])/C # 
            #print(deltathe*np.exp((v[i,t]-vth[i,t])/deltathe)/taue)
        
            v[i,t+1] = v[i,t] + dv*dt
            if v[i,t +1] > vpeak:
                w[i,t+1] = w[i,t+1] + bw_adapt
                vth[i,t+1] = vth[i,t] + ath
                #print(tt,v[i,t+1])
                spiked = "true"
                lastspiked[i] = tt 
                #spikeda.append(lastspiked)
                v[i,t+1] = vre
        #if v[i,t+1] < vre:
        #   v[i,t+1] = vre
            
        
        
        
#print(spikeda) 
spike = np.ones(len(spikeda))    
plt.plot(tarray , v[0,:])   
plt.plot(tarray , vth[0,:])
plt.plot(tarray , w[0,:])
#plt.plot(spikeda, spike, 'o')
plt.show()

    