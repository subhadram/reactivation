#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:10:00 2025

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


TR = str(1)
dt = 1e-4           # dt in sec
tmax = 20         # maximum time in sec
tvector = np.arange(0,tmax,dt)
allrates = np.zeros((3,len(tvector)))
EP =str(4)
eps =int(0)
with open('allratesuc_{0}.pkl'.format(TR),'rb') as f: 
    allrates = np.squeeze(pickle.load(f,encoding = 'latin1'))
    

rateg1 = allrates[0,:]
rateg2 = allrates[1,:]
rateov = allrates[2,:]



print(np.mean(rateg1),np.std(rateg1))
print(np.mean(rateg2),np.std(rateg2))
print(np.mean(rateov),np.std(rateov))

with open('allratesuc055_{0}.pkl'.format(TR),'rb') as f: 
    allrates = np.squeeze(pickle.load(f,encoding = 'latin1'))
    

rateg1 = allrates[0,:]
rateg2 = allrates[1,:]
rateov = allrates[2,:]


print(np.mean(rateg1),np.std(rateg1))
print(np.mean(rateg2),np.std(rateg2))
print(np.mean(rateov),np.std(rateov))

thigh1 = np.mean(rateg1 > 10)
thigh2 = np.mean(rateg2 > 10)
#print([i for i,x in enumerate(rateov) if rateg1[i] <= 10 and rateg2[i]<=10])
thighli = [i for i,x in enumerate(rateov) if rateg1[i] <= 10 and rateg2[i]<=10]
thighhi = [i for i,x in enumerate(rateov) if rateg1[i] > 10 or rateg2[i]>10]

thighl = rateov[thighli]
thighh = rateov[thighhi]
inpts = rateg1>10 

timevec = tvector[1:]
prod1 = inpts[1:]!=inpts[:-1]
prod1 = np.where(prod1==1)

prod11 = timevec[inpts[1:]!=inpts[:-1]]
prod12 = np.diff(prod11)

event_indices = []
if len(prod12)>0 and inpts[prod1[0][0]] == False:
    #print("Hi")
    lencs = len(prod12[::2])
    prod13 = np.mean(prod12[::2])
    event_indices = prod1[0][::2]
    #print(prod13,"1")
elif len(prod12)>0 and inpts[prod1[0][0]] == True:
    prod13 = np.mean(prod12[1::2])
    lencs = len(prod12[1::2])
    event_indices = prod1[0][1::2]
else:
    prod13 = 0.0
    lencs = 0

    
inpts2 = rateg2>10 
#timevec = tvector[1:]
prod2 = inpts2[1:]!=inpts2[:-1]
prod2 = np.where(prod2==1)
#print(prod2)
prod21 = timevec[inpts2[1:]!=inpts2[:-1]]
prod22 = np.diff(prod21)
#print(inpts[prod1[0][0]])
event_indices2 = []
if len(prod22)>0 and inpts[prod2[0][0]] ==False :
    lenis = len(prod22[::2])

    prod23 = np.mean(prod22[::2])
    event_indices2 = prod2[0][::2]
    
elif len(prod22)>0 and inpts[prod2[0][0]] == True:
    lenis = len(prod22[1::2])
    event_indices2 = prod2[0][1::2]

   
    prod23 = np.mean(prod22[1::2])
else:
    lenis = 0
    prod23 = 0.0


timescs = timevec[event_indices]
timesis = timevec[event_indices2]
print(len(timescs),len(timesis))
difft = []

A= timescs
B =timesis
AnA=[]
AnB=[]
for i in range(len(A)-1):
    j=0
    while j < len(B):
        if (A[i] < B[j]):
            if (A[i+1] > B[j]):
                AnB.append(B[j]-A[i])
                j = len(B)
            else:
                AnA.append(A[i+1]-A[i])
                j = len(B)
        else:
            j=j+1
            

BnA=[]
BnB=[]
for i in range(len(B)-1):
    j=0
    while j < len(A):
        if (B[i] < A[j]):
            if (B[i+1] > A[j]):
                BnA.append(A[j]-B[i])
                j = len(A)
            else:
                BnB.append(B[i+1]-B[i])
                j = len(A)
        else:
            j=j+1

print(len(AnA), len(AnB))
                
print(len(BnA), len(BnB))

from scipy.stats import ks_2samp

print(ks_2samp(BnA, BnB))






"""
for csi in range(len(timescs)):
    for isi in range(len(timesis)):
        difft.append(timescs[csi] - timesis[isi])
        
figh = plt.figure(1000+eps)

levent = len(event_indices)
levent2 = len(event_indices2)
timescsr = np.random.uniform(0,tmax,levent)
timesisr = np.random.uniform(0,tmax,levent2)
difftr = []
for csir in range(len(timescs)):
    for isir in range(len(timesis)):
        difftr.append(timescsr[csir] - timesisr[isir])


plt.hist(difft, alpha = 0.5,label = "time difference distribution")
plt.hist(difftr,alpha = 0.5, label = "time difference distribution for uniform distribution")
plt.legend()
plt.show()
plt.close(figh)

from scipy.stats import ks_2samp

print(ks_2samp(difft, difftr))
"""

