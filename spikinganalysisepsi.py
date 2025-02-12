#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:31:20 2023

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

with open('firingratesrec2.pkl','rb') as f:
	firingrates = pickle.load(f,encoding = 'latin1')


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
#firingrates = np.zeros((len(aa),len(epsilon),trials,3,len(tvector)))
fracti = np.zeros((len(aa),trials))
fractc = np.zeros((len(aa),trials))
maxratecs = np.zeros((len(aa),trials))
maxrateis = np.zeros((len(aa),trials))
spikescs = np.zeros((len(aa),trials))
spikesis = np.zeros((len(aa),trials))
for k in range(len(aa)):   
    
    for i in range(len(epsilon)):
        
        for j in range(trials):
            a = aa[k]
            
           
            epsi = epsilon[i]
            EP = str(epsi)
            J = np.zeros((N,N))
            Jee1 = a
            Jee2 = a + epsi
            AS = aa[k]
            spikesg1= firingrates[k,i,j,0,:] 
            spikesg2= firingrates[k,i,j,1,:]
            spikesov = firingrates[k,i,j,2,:]
            rateg1 = spikes_to_rate(spikesg1, dt, 0.05)
            rateg2 = spikes_to_rate(spikesg2, dt, 0.05)
            rateov = spikes_to_rate(spikesov, dt, 0.05)
            # fig = plt.figure(1)
            # plt.plot(tvector, rateg1)
            # plt.plot(tvector, rateg2)
            # plt.plot(tvector, rateov)
            # plt.title(r" $\epsilon$ = "+ EP + "a=" + "AS")
            # plt.savefig('frate_%s.png'%(EP))
            # plt.close(fig)
            maxratecs[k,j] =  np.max(rateg1[1000:])
            maxrateis[k,j] =  np.max(rateg2[1000:])
            b = (rateg2-rateg1)
            b1 = (rateg1 - np.mean(rateg1))/np.std(rateg1)
            b2 = (rateg2 - np.mean(rateg2))/np.std(rateg2)
            fractinis = np.where(rateg1> 10.0, 1,0 )
            fractincs = np.where(rateg2> 10.0, 1,0 )
            fracti[k,j] = np.mean(fractinis)
            fractc[k,j] = np.mean(fractincs)
            spikescs[k,j] = np.sum(spikesg1[1000:])
            spikesis[k,j] =  np.sum(spikesg2[1000:])
        
            
            #plt.title(r"g =" + GG + r" $\epsilon$ = "+ EP + " a =" + AA + r" $\sigma_x$ = " + NAMP + " tr = " + TR)
           # plt.savefig('xx_g_%s_ep_%s_a_%s_ic_%s_na_%s_tr_%s.png'%(GG, EP,AA,LL,NAMP,TR))
           # plt.close(fig)
           
           
plt.errorbar(aa,np.mean(fractc,1), yerr = np.std(fractc,1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5, label = "CS")
plt.errorbar(aa,np.mean(fracti,1),yerr = np.std(fracti,1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5, label = "IS")
plt.xlabel(r"recurrent weight")
plt.ylabel("fraction of time in state")
#plt.xlim(0,0.35)
plt.legend()
plt.show()

plt.errorbar(aa,np.mean(spikescs,1), yerr = np.std(spikescs,1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5, label = "CS")
plt.errorbar(aa,np.mean(spikesis,1), yerr = np.std(spikesis,1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,  label = "IS")
plt.xlabel(r"recurrent weight")
plt.ylabel("number of spikes per neuron")
#plt.xlim(0,0.35)
plt.legend()
plt.show()

plt.errorbar(aa, np.mean(maxratecs,1), yerr = np.std(maxratecs,1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5, label = "CS")
plt.errorbar(aa, np.mean(maxrateis,1),yerr = np.std(maxrateis,1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5, label = "IS")
plt.xlabel(r"overlap")
plt.ylabel("maximum firing rate of the ensemble")
#plt.xlim(0,0.35)
plt.legend()
plt.show()