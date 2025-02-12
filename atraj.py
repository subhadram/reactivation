#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:34:19 2024

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib
import pickle
from scipy.optimize import curve_fit


dt = 1e-4
epsilon = np.arange(1.0e-9,1.1e-9, 0.1e-9)
w0array = np.arange(0e-9,2.1e-9, 0.2e-9)
warray = np.arange(1.5e-9, 2.51e-9,1.0e-10)
#warray = np.arange(0,0.31 ,0.01)
#warray = np.arange(4, 41, 4)
#warray = np.logspace(-5,5, num =11)
print(warray)
trials = 100
xdata = epsilon
csfrac = np.zeros((len(warray),100))
isfrac = np.zeros((len(warray),100))
for tr in range(1,101):
    TR = str(tr)
    with open('csfracao_{0}.pkl'.format(TR),'rb') as f: 
        csfrac[:,tr-1] = np.squeeze(pickle.load(f,encoding = 'latin1'))
    with open('isfracao_{0}.pkl'.format(TR),'rb') as f:
        isfrac[:,tr-1] = np.squeeze(pickle.load(f,encoding = 'latin1'))
        

    #with open('isfraca.pkl','rb') as f:
	#isfrac = pickle.load(f,encoding = 'latin1')
    
#print(isfrac.shape)
# import scipy.io as sio
# csfrac2 = sio.loadmat('csfraca.mat')
# csfrac = csfrac2['csfrac']

# isfrac2 = sio.loadmat('isfraca.mat')
# isfrac = isfrac2['isfrac']


# def sigmoid(x, a,b,c):
#     y = a / (1 + np.exp(-b*(x))) + c
#     return (y)
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)
def linearl(x, m, c):
    y = m*x+c
    return y

slopes = np.zeros(len(warray))
slopes1 = np.zeros(len(warray))
xdata = 10.0**10*epsilon
timespent = np.zeros((len(warray),len(warray)))

plt.rcParams.update({'font.size': 12})
plt.errorbar(warray, np.mean(csfrac[:,:],1), yerr =np.std(csfrac[:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "forestgreen", label = "CS")
plt.errorbar(warray, np.mean(isfrac[:,:],1), yerr =np.std(isfrac[:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "gold", label = "IS")
plt.xlabel(r"$\epsilon$")
plt.ylabel("fraction of time in state")
plt.legend()
#plt.ylim(0,0.08)
plt.show()

timea = np.arange(0,10000,0.1)
def expo(t,k,a0,ab):
    y = 25e-10*(0.1*np.exp(-k*t) + 0.3*np.exp(-k*t/8.0) )+ ab 
    return y
expon = expo(timea, 0.0005, 0.01, 1.5e-9)
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

plt.plot(timea, expon)
plt.show()

p1 = [0.2, 0.2,0.0001, 0.2]
popt, pcov = curve_fit(sigmoid, 10**10*warray, np.mean(csfrac[:,:],1),p1, method='dogbox')

plt.plot(warray,np.mean(csfrac[:,:],1),'o')
plt.plot(warray, sigmoid(10**10*warray, *popt))
plt.show()

trajc = sigmoid(10**10*expon,*popt)
plt.plot(timea,trajc)
plt.show()

