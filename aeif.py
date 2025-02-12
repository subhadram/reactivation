#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:44:08 2023

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.special import erf
#import matplotlib
#import pickle

Ni = 10
Ne = 40
jee = 0.2 # initial e to e weight
jei = 0.1 #initial e to i connectivity
jie = -0.1 # fixed i to e connectivity
jii = -0.1 # fixed i to i connectivity
Ncells = Ni + Ne
p = 0.1

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
tarray = np.arange(0,100,dt)
gl = 10.0
#Ncells =100


V = np.zeros((len(tarray),Ncells))
vth = np.zeros((len(tarray),Ncells))
w = np.zeros((len(tarray),Ncells))
rx = np.zeros((len(tarray),Ncells))
nextx = np.zeros((len(tarray),Ncells))

I = 0
tr = np.zeros(Ncells)
V[0] = -60+ np.random.randn(Ncells)
w[0] = np.random.randn(Ncells)
vth[0] = np.random.randn(Ncells)
vpeak = 20.0


taue = 20 #e membrane time constant
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
vre = -60 #reset potential
taurefrac = 1 #absolute refractory period
aw_adapt = 4 #adaptation parameter a
bw_adapt = .805 #adaptation parameter b
tauw_adapt = 150 #adaptation timescale

#connectivity
Ncells = Ne+Ni
tauerise = 1 #e synapse rise time
tauedecay = 6 #e synapse decay time
tauirise = .5 #i synapse rise time
tauidecay = 2 #i synapse decay time
rex = 4.5 #external input rate to e (khz)
rix = 2.25 #external input rate to i (khz)

jeemin = 1.78 #minimum ee strength
jeemax = 21.4 #maximum ee strength

jeimin = 48.7 #minimum ei strength
jeimax = 243 #maximum ei strength

jex = 1.78 #external to e strength
jix = 1.27 #external to i strength

#voltage based stdp
altd = .0008 #ltd strength
altp = .0014 #ltp strength
thetaltd = -70 #ltd voltage threshold
thetaltp = -49 #ltp voltage threshold
tauu = 10 #timescale for u variable
tauv = 7 #timescale for v variable
taux = 15 #timescale for x variable

#inhibitory stdp
tauy = 20 #width of istdp curve
eta = 1 #istdp learning rate
r0 = .003 #target rate (khz)

#populations
Npop = 2
pmem = 0.05
Nmaxmembers = 300
#Npop = size(popmembers,1) #number of assemblies
#Nmaxmembers = size(popmembers,2) #maximum number of neurons in a population

#simulation
#dt = .1 #integration timestep
T = 100 #simulatiogkn time
Nskip = 1000 #how often (in number of timesteps) to save w_in
vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset
dtnormalize = 20 #how often to normalize rows of ee weights
stdpdelay = 1000 #time before stdp is activated, to allow transients to die out
Nspikes = 100 #maximum number of spikes to record per neuron



def structinitJ():
    J = np.zeros((Ncells, Ncells))
    #setting up initial weights
    J[0:Ne, 0:Ne] = jee
    J[0:Ne, Ne:Ncells] = jie
    J[Ne:Ncells, 0:Ne] = jei
    J[Ne:Ncells, Ne:Ncells] = jii
    strc = np.array(np.where(np.random.rand(Ncells, Ncells) < p,1,0))  
    J = np.multiply(J,strc)  
    return J

def make_self_connections_zero(J):
    for i in range(0,len(J)):
        J[0,0] = 0.0
    return J


def sim():
    rx = np.zeros(Ncells)
    nextx = np.zeros(Ncells)
    last_spike = np.zeros(Ncells)
    spiked= np.zeros(Ncells, dtype = bool)
    forwardinputseprev = np.zeros(Ncells)
    forwardinputsiprev = np.zeros(Ncells)
    lastspike = np.zeros(Ncells)
    xerise = np.zeros(Ncells)
    xedecay =  np.zeros(Ncells)
    xirise = np.zeros(Ncells)
    xidecay =  np.zeros(Ncells)
    
    for i in range(0, Ncells):
        V[0,i] = vre + (vth0-vre)*np.random.rand()
        rx[i] = rex
        nextx[i] = -np.log(1-np.random.rand()/rx[i])

    
    for t in range(1,len(tarray)-1):
        
        tt = dt*t
        
        for i in range(0,Ncells):
            
        
            dw = (aw_adapt*(V[t-1,i] - vleake) - w[t-1,i])*dt/tauw_adapt
           
            w[t,i] = w[t-1,i] + dw
        
            dvth = (vth0 - vth[t-1,i])*dt/tauth
        
            vth[t,i] = vth[t-1,i] + dvth
        
            if tt > (lastspike[i] + taurefrac):
                dv = (vleake - V[t-1,i])/taue + deltathe*np.exp((V[t-1,i]-vth[t-1,i])/deltathe)/taue   - w[t-1,i]/C + I/C 
                #(vleake - v[cc] + deltathe*exp((v[cc]-vth[cc])/deltathe))/taue + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C - wadapt[cc]/C;
                V[t,i] += dt*dv
                if V[t,i] > vpeak:
                    spiked[i] = "true"
                    w[t,i] = w[t-1,i] + bw_adapt
                    vth[t,i] = vth[t-1,i] + ath
                    
                    
                    
                if spiked[i]:
                    spiked[i] = "true"
                    V[t,i] = vre
                    lastspike[i] = tt
                
            
         
    return V
            
       

        
        
V = sim()        
#print(V)
#plt.plot(tarray, vth)
#plt.plot(tarray, V[:,0])
plt.plot(tarray, V[:,0])
#plt.plot(tarray,w )
plt.show()