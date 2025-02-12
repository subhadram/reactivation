#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 19:15:23 2023

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib
import pickle

Ne = 400
Ni = 100
jee = 0.2 # initial e to e weight
jei = 0.1 #initial e to i connectivity
jie = -0.1 # fixed i to e connectivity
jii = -0.1 # fixed i to i connectivity
Ncells = Ni + Ne
p = 0.1

stim = np.zeros((4,4))



#membrane dynamics
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
rex = 1.0 #external input rate to e (khz)
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
pmem = 0.1
I = 300
Nmaxmembers = 300
#Npop = size(popmembers,1) #number of assemblies
#Nmaxmembers = size(popmembers,2) #maximum number of neurons in a population

#simulation
dt = .1 #integration timestep
T =1000 #simulatiogkn time
Nskip = 1000 #how often (in number of timesteps) to save w_in
vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset
dtnormalize = 20 #how often to normalize rows of ee weights
stdpdelay = 1 #time before stdp is activated, to allow transients to die out
Nspikes = 20000 #maximum number of spikes to record per neuron



def structinitJ():
    J = np.zeros((Ncells, Ncells))
    #setting up initial weights
    J[0:Ne, 0:Ne] = jee
    J[0:Ne, Ne:Ncells] = jei
    J[Ne:Ncells, 0:Ne] = jie
    J[Ne:Ncells, Ne:Ncells] = jii
    strc = np.array(np.where(np.random.rand(Ncells, Ncells) < p,1,0))  
    J = np.multiply(J,strc)  
    return J

def make_self_connections_zero(J):
    for i in range(0,len(J)):
        J[0,0] = 0.0
    return J


# y = structinitJ()
# y = make_self_connections_zero(y)
# print(y[9,9])

#print(np.where(np.random.rand(Ncells, Ncells) < p, 1,0))



def makegroups(Npop,pmem,Nmaxmembers):
    popmembers = -1*np.ones((Npop, Nmaxmembers))
    patterns = np.zeros((Npop, Ncells))
    for i in range(0,Npop):
        members = np.array(np.argwhere(np.random.rand(Ne) < pmem))
        print(members.shape)
        popmembers[i,0:len(members)] = np.squeeze(members)
        patterns[i,members] = 1
    #print(popmembers)
    return popmembers, patterns

    

def sim(weights,popmembers,rx):
    times = np.zeros((Ncells, Nspikes)) 
    ns = np.zeros(Ncells,dtype = int)
    forwardinputse = np.zeros(Ncells) #summed weight of incoming E spikes
    forwardinputsi = np.zeros(Ncells)  #summed weight of incoming I spikes
    forwardinputseprev = np.zeros(Ncells)  #summed weight if incoming E spikes, prev time step
    forwardinputsiprev = np.zeros(Ncells) # summed weight if incoming I spikes, prev time step
    Vrec = np.zeros((Ncells,int(T/dt)))
    spikedt =  np.zeros((Ncells,int(T/dt)))
    Vrec = np.zeros((Ncells,int(T/dt)))
    recx =  np.zeros((Ncells, int(T/dt)))
    weightr = np.zeros((Ncells, Ncells, int(T/dt) ))
    #exponential variables
    xerise = np.zeros(Ncells)
    xedecay =  np.zeros(Ncells)
    xirise = np.zeros(Ncells)
    xidecay = np.zeros(Ncells)
    
    v = np.zeros(Ncells)
    nextx = np.zeros(Ncells) #time of next excitatory input
    sumwee0 = np.zeros(Ne) #initial summed e weight
    #rx = np.zeros(Ncells) #external rate
    Nee= np.zeros(Ncells, dtype = int)#number of e->e inputs
    
    #initial values for the variables.
    for i in range(0, Ncells):
        
        v[i] = vre + (vth0-vre)*np.random.rand()
        if i <Ne:
            #rx[i] = rex
            nextx[i] = -np.log(1-np.random.rand()/rex)
            
            sumwee0[i] = np.sum(weights[i,0:Ne])
            Nee[i]= np.sum(1.*np.where(weights[i,0:Ne]>0.0, 1, 0))
        else:
            #rx[i] = rix
            nextx[i] = -np.log(1-np.random.rand()/rex)
    vth = vth0*np.ones(Ncells)
    wadapt = aw_adapt*(vre - vleake)*np.ones(Ncells)
    lastspike= -100*np.ones(Ncells)
    trace_istdp = np.zeros(Ncells)
    u_vstdp = np.zeros(Ne)
    v_vstdp = np.zeros(Ne)
    x_vstdp = np.zeros(Ne)
    
    Nsteps = int(T/dt)
    
    inormalize = int(dtnormalize/dt)
    print(Nsteps, inormalize)
    
    for tt in range(0, Nsteps):
        Vrec[:,tt] = v
        #recx[:,tt] = rx
        weightr[:,:,tt] = weights
        #print(weights)
       
        t = dt*tt
        #print(t)
        forwardinputse[:] = 0.0
        forwardinputsi[:] = 0.0
        tprev = dt*(tt-1) #only needed for the choose the simulation period.
        """
        for ss in range(0, len(stim[:,0])):
            #print(ss)
            if (tprev<stim[ss,1]) & (t>=stim[ss,1]): #entering stimulation period
                print('hi')
                ipop = int(stim[ss,0])
                for ii in range(0, Nmaxmembers):
                    #print(popmembers[ipop,ii])
                    if popmembers[ipop,ii] == -1:
                        #print("Hi")
                        break
                    rx[int(popmembers[ipop,ii])] += stim[ss,3] #exciting stimulation period
            if (tprev<stim[ss,2]) & (t>=stim[ss,2]):
                ipop = int(stim[ss,0])
                for ii in range(0, Nmaxmembers):
                    if int(popmembers[ipop,ii]) == -1:
                        break
                    rx[int(popmembers[ipop,ii])] -= stim[ss,3]
                    
            """
        
        
        
        spiked= np.zeros(Ncells, dtype = bool)
        for cc in range(0,Ncells):
            trace_istdp[cc] -= dt*trace_istdp[cc]/tauy
            while(t > nextx[cc]):
                nextx[cc] += -np.log(1-np.random.rand()/rx[i,tt]) #random time generation for event based simulations
                print(nextx[cc])
                if cc < Ne:
                    forwardinputseprev[cc] += jex
                else:
                    forwardinputseprev[cc] += jix
                    
            xerise[cc] += -dt*xerise[cc]/tauerise + forwardinputseprev[cc]
            xedecay[cc] += -dt*xedecay[cc]/tauedecay + forwardinputseprev[cc]
            xirise[cc] += -dt*xirise[cc]/tauirise + forwardinputsiprev[cc]
            xidecay[cc] += -dt*xidecay[cc]/tauidecay + forwardinputsiprev[cc]
            
            if cc < Ne: #adaption exponential and threshold and stdp updates
                vth[cc] += dt*(vth0 - vth[cc])/tauth
                wadapt[cc] += dt*(aw_adapt*(v[cc]-vleake) - wadapt[cc])/tauw_adapt
                u_vstdp[cc] += dt*(v[cc] - u_vstdp[cc])/tauu
                v_vstdp[cc] += dt*(v[cc] - v_vstdp[cc])/tauv
                x_vstdp[cc] -= dt*x_vstdp[cc]/taux
                
            if t > (lastspike[cc] + taurefrac):
                ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise)
                gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise)
                
                
                if cc < Ne:
                    dv = (vleake - v[cc] + deltathe*np.exp((v[cc]-vth[cc])/deltathe))/taue - wadapt[cc]/C + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C 
                    v[cc] += dt*dv
                    if v[cc] > vpeak:
                        spiked[cc] = "true"
                        wadapt[cc] += bw_adapt
                else:
                    dv = (vleaki - v[cc])/taui + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C
                    v[cc] += dt*dv
                if v[cc] > vth0:
                    spiked[cc] = "true"
                    
            if spiked[cc]:
                v[cc] = vre
                lastspike[cc] = t
                ns[cc] += 1
                if ns[cc] < Nspikes:
                    times[cc,ns[cc]] = t
                if cc < Ne:
                    vth[cc] = vth0 + ath
                
               
                forwardinputse[cc] = forwardinputse[cc] + np.sum(weights[cc,0:Ne])
                forwardinputsi[cc] = forwardinputsi[cc] + np.sum(weights[cc,Ne:Ncells])
                    

        spikedt[:,tt] = spiked                             
        forwardinputseprev = forwardinputse
        forwardinputsiprev = forwardinputsi
        

        
        
    #print(spikedt)
                            
                        
                
        
                    
                
            
        
        
                            
                        
                
                
                
            
        
    
    return Vrec,spikedt,times, ns, weightr

import scipy.stats

def spike_to_rate(spikes, window_std=10):
    window_size = np.arange(-3*window_std,3*window_std,1)
    window = scipy.stats.norm.pdf(window_size, 0, window_std)
    window /= window.sum()
    n_units = spikes.shape[0]
    estimate = np.zeros_like(spikes) # Create an empty array of the same size as spikes
    for i in range(n_units):
        y = np.convolve(window, spikes[i,:], mode='same')
        estimate[i,:] = y
    return estimate
N = Ncells
J = np.zeros((N,N))
Jee1 = 2
Jee2 = 2.6
Jei = -1.2
Jie = 1.2
overlap = 0.0
group1 = 0.45
group2 = 0.45
inhgrp = 0.1

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

groups, patterns = makegroups(Npop,pmem,Nmaxmembers)
groups = groups.astype(int)
#print(groups[1])
rx = 2.0*np.ones((Ncells,int(T/dt)))
#print(patterns)
print(groups[0])

#rx[groups[0],10000:20000] = 2.5
#rx[groups[0],30000:40000] = 2.5
#rx[groups[0],50000:60000] = 2.5
#rx[groups[0],90000:100000] = 2.5
# rx[groups[0],130000:140000] = 4.0
# rx[groups[0],170000:180000] = 4.0
# rx[groups[0],210000:220000] = 4.0
# rx[groups[0],250000:260000] = 4.0
# rx[groups[0],290000:300000] = 4.0
# rx[groups[0],330000:340000] = 4.0
# rx[groups[0],370000:380000] = 4.0
# rx[groups[0],410000:420000] = 4.0
# rx[groups[0],450000:460000] = 4.0
# rx[groups[0],490000:500000] = 4.0
# rx[groups[0],530000:570000] = 4.0
# rx[groups[0],600000:610000] = 4.0

V, spikes, times, ns, weightst = sim(J,groups,rx)

print(V.shape)
#plt.plot(V[0,:])
plt.imshow(weightst[:,:,0])
#plt.imshow(weightst[:,:,:-1])
plt.show()
plt.imshow(weightst[:,:,10])
#plt.imshow(weightst[:,:,:-1])
plt.show()
plt.imshow(weightst[:,:,-1])
#plt.imshow(weightst[:,:,:-1])
plt.show()
# plt.imshow(1.*spikes,aspect = "auto")
# plt.show()

# rate= spike_to_rate(1.*spikes)

# plt.imshow(rate)
# plt.show()


# overl = np.dot(rate.transpose(), patterns[0])
plt.imshow(rx,aspect = "auto")
plt.show()

plt.plot(V[groups[0, 0],:])
plt.show()

plt.imshow(spikes)
plt.show()

rowcount = 0
for pp in range(Npop-1):
    for cc in range(Ne):
        
        rowcount+=1
        ind =   cc
        #idx2 = np.arange(0,ns[ind],1)
        #idx2 = idx2.astype("int")
        vals = times[ind,0:ns[ind]]
        y = rowcount*np.ones(len(vals))
        plt.scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)

plt.show()
