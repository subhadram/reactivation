#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:30:02 2023

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib
import pickle

N= 500
p = 0.1

J = np.zeros((N,N))

gp1 = 0.4
gp2 = 0.4
gp1i = int(gp1*N)
overlapp = 0.1
overlapi = gp1i - int(overlapp*N)

gp2i = gp1i + int(gp2*N)-int(overlapp*N)


def phi(xx,mu,sig):
		fy = 1/2.0*(1+erf((xx-mu)/(np.sqrt(2)*sig)))
		return fy

group1 = np.where(np.random.rand(N) < p,1,0)  
group11 = np.argwhere(group1 > 0)
print(group1)
print(group11)
Jee1 = 10.0
Jee2 = 60.0

# print(len(group11))
# for i in range(len(group11)):
#     for j in range(len(group11)):
#         #print(group11[j], group11[i])
#         J[group11[j], group11[i]] = Jee1
Ninh = 500 - int(gp1*N*2)     
J[0:gp1i, 0:gp1i] = Jee1
J[overlapi:gp2i,overlapi:gp2i] = Jee2
randomex1 = 3 + 1.0*np.random.normal(0,1,[150,350])
#randomex2 = 1 + 2.0*np.random.normal(0,1,[150,N])
randominh1 =  -3 + 1.0*np.random.normal(0,1,[N,150])

randominh1 = np.where(randominh1 > 0 , 0, randominh1)
randomex1 = np.where(randomex1 < 0 , 0, randomex1 )
#randomex2 = np.where(randomex2 < 0 , 0, randomex2 )
J[gp2i:,:gp2i] =  randomex1
J[:,gp2i:] = randominh1

strc = np.array(np.where(np.random.rand(N, N) < p,1,0))  
J = np.multiply(J,strc)  


plt.imshow(J)
plt.show()
time = 3000
r = np.zeros((N,time))
mu = 3.0,
sig = 0.1
dt = 0.1
tau = 10
tause = 15
tausi = 20
I = np.random.normal(0,1,[N,time])
tarray = np.arange(0,time,dt)
V_Thresh = -50
deltaT =2
G_L = 1.2
E_L = -52.0
C = 200
a = 4.2
tauw = 100
Vth = 0
Vreset = -60
Vpeak = 20
bw = 0.1
# # #print(X)
# for t in range(0,time-1):
#     r[:,t+1]=  r[:,t] + (-r[:,t] + phi(J.dot(r[:,t])+ I[:,t],mu,sig))*dt/tau
# for i in range(200):    
#     plt.plot(r[i,:])

# plt.show()


se = np.zeros(350)        

si =  np.zeros(150)

s = np.zeros(N)
v =  np.zeros((N, len(tarray)))
w = np.zeros((N, len(tarray)))
G_ahp = np.zeros((N, len(tarray)))
spikeso = np.zeros((N, len(tarray)))
v[:,0] = np.random.normal(0,1,[N])-60
I = np.random.normal(0,1000,[N, len(tarray)])
E_K = -50

for tt in range(len(tarray)-1):
    se = se*np.exp(-dt/tause)
    si = si*np.exp(-dt/tausi)
    s[0:350] = se
    s[350:] = si
    v[:,tt+1] = v[:,tt] + dt*( G_L*(E_L-v[:,tt] + deltaT*np.exp((v[:,tt]-V_Thresh)/deltaT) ) - w[:,tt]+I[:,tt] +np.dot(J,s))/C   
    w[:,tt+1] = w[:,tt] + dt*( a*(v[:,tt]-E_L) - w[:,tt] )/tauw;
    cellsfired = np.argwhere(v[:,tt] >= Vth)
    #print(cellsfired)
    spikeso[cellsfired, tt+1] = 1.0
    #G_ahp[cellsfired, tt+1] = 400
    w[:,tt+1] += bw
    v[cellsfired, tt+1] = Vreset
    v[cellsfired, tt] = Vpeak
    #negs = np.argwhere(v[:,tt] <= -80)
    #v[negs, tt+1] = -80
    
    
    
    
    
    
    
    
 
plt.plot(v[0,:])
plt.plot(v[100,:])
plt.show()
    
plt.imshow(spikeso,aspect = "auto",cmap="gist_gray")
plt.show()
        
    
    
