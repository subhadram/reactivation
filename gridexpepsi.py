#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:07:43 2023

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

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
tmax = 20          # maximum time in sec
tvector = np.arange(0,tmax,dt)




def spikes_to_rate(spikes, dt, window):
    windowsize = int(window/dt)
    sumspike = np.zeros_like(spikes)
    i = 0
    while i < len(spikes):
        sumspike[i:i+windowsize] = np.sum(spikes[i:i+windowsize])
        i = i + windowsize
        
    sumspike = sumspike/(window)
    return sumspike
        
trials = 10

aa = np.arange(1.0e-9,3.1e-9,2e-10)
epsilon = np.arange(0.5e-9, 1.0e-9, 5e-10)
firingrates = np.zeros((len(aa),len(epsilon),trials,3,len(tvector)))

    
for k in range(len(aa)):   
    
    for i in range(len(epsilon)):
        
        for j in range(trials):
            a = aa[k]
            
           
            epsi = epsilon[i]
            EP = str(epsi)
            J = np.zeros((N,N))
            Jee1 = a
            Jee2 = a + epsi
            Jei = 5e-9
            Jie = 2.5e-9
            overlap = 0.0
            group1 = 0.4
            group2 = 0.4
            dw = 3e-8
            ds = 1
            group1ei = int((group1)*N)
            group1ei = int((group1)*N)
            group2si = int((group1-overlap)*N)
            group2ei = int((group1+group2- overlap)*N)
            eind = np.zeros(N)
            eind[0:group2ei] = int(1)
            iind = np.zeros(N)
            iind[group2ei:] = int(1)
            inhind = int(0.8*N)
            J[inhind:,:inhind] = Jie
            J[:,inhind:] = Jei
            J[:group1ei,:group1ei] = Jee1
            J[group2si:group2ei,group2si:group2ei ] =  Jee2
            p = 0.1
            strc = np.array(np.where(np.random.rand(N, N) < p,1,0))
            J = np.multiply(J,strc)
            lamda = 0.0
            I = 15e-8*np.sqrt(dt)*(lamda*np.random.normal(0,1,[len(tvector)])+ np.sqrt(1- lamda**2)*np.random.normal(0,1,[N,len(tvector)]))
            #I[200:400,:] = 0.0
            #I[:,10000:] = 0.0
            V= np.zeros((N,len(tvector)))
            s= np.zeros((N,len(tvector)))
            w =  np.zeros((N,len(tvector)))
            spikes = np.zeros((N,len(tvector)))
            
            E_E = 0
            E_I = -70e-3
            V[:,0] = E_L + (V_Thresh-E_L)*np.random.rand(N)
            w[:,0] = dw*np.random.rand(N)
            s[:,0] = ds*np.random.rand(N)
            inhindex = np.zeros(N)
            inhindex[0:inhind] = 1.0
            
            for tt in range(len(tvector)-1):
                cellsfired = np.argwhere(V[:,tt] >= Vmax)
                cellsfirede = cellsfired[cellsfired< group2ei]
                cellsfiredi = cellsfired[cellsfired>= group2ei]
                #print(cellsfirede)
                V[cellsfired,tt] = V_Reset
                w[cellsfired,tt] = w[cellsfired,tt] + dw
                s[cellsfired,tt] = s[cellsfired,tt] + ds
                spikes[cellsfired,tt] =  1
                inputsfrome = np.matmul(J[:,:inhind],s[:inhind,tt])
                inputsfromi = np.matmul(J[:,inhind:],s[inhind:,tt])
                V[:,tt + 1 ] = V[:,tt] + dt*( G_L*(E_L-V[:,tt] + deltaT*np.exp((V[:,tt]-V_Thresh)/deltaT) )+ inhindex*w[:,tt]*(E_K-V[:,tt]) + I[:,tt]  + inputsfrome*(E_E - V[:,tt]) + inputsfromi*(E_I - V[:,tt]))/C
                s[:,tt+1] = s[:,tt]*np.exp(-dt/taus)
                w[:,tt+1] = w[:,tt]*np.exp(-dt/tauw)
            
            spikesg1 = np.mean(spikes[:group2si,:],0)
            spikesg2 = np.mean(spikes[group2si+int(overlap*N):group2ei,:],0)
            spikesov = np.mean(spikes[group2si:group2si+int(overlap*N),:],0)
            firingrates[k,i,j,0,:] = spikesg1
            firingrates[k,i,j,1,:] = spikesg2
            firingrates[k,i,j,2,:] = spikesov
            # rateg1 = spikes_to_rate(spikesg1, dt, 0.05)
            # rateg2 = spikes_to_rate(spikesg2, dt, 0.05)
            # rateov = spikes_to_rate(spikesov, dt, 0.05)
            # fig = plt.figure(1)
            # plt.plot(tvector, rateg1)
            # plt.plot(tvector, rateg2)
            # plt.plot(tvector, rateov)
            # plt.title(r" $\epsilon$ = "+ EP)
            # plt.savefig('frate_%s.png'%(EP))
            # plt.close(fig)
            # maxratecs[i,j] =  np.max(rateg1[1000:])
            # maxrateis[i,j] =  np.max(rateg2[1000:])
            # b = (rateg2-rateg1)
            # fractinis = np.where(b> 2.0, 1,0 )
            # fractincs = np.where(b< -2.0, 1,0 )
            # spikescs[i,j] = np.sum(spikes[:group2si,1000:])
            # spikesis[i,j] =  np.sum(spikes[group2si+int(overlap*N):group2ei,1000:])
        
            
            #plt.title(r"g =" + GG + r" $\epsilon$ = "+ EP + " a =" + AA + r" $\sigma_x$ = " + NAMP + " tr = " + TR)
           # plt.savefig('xx_g_%s_ep_%s_a_%s_ic_%s_na_%s_tr_%s.png'%(GG, EP,AA,LL,NAMP,TR))
           # plt.close(fig)

pickle.dump( firingrates, open( "firingratesrec2.pkl","wb"))
