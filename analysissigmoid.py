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
epsilon = np.arange(-0.2e-9,0.21e-9, 0.1e-9)
w0array = np.arange(1e-9,2.1e-9, 0.1e-9)
warray = np.arange(0.0e-9, 11e-9, 1e-9)

xdata = epsilon
csfrac = np.zeros((len(warray),len(epsilon),100))
with open('csfraca.pkl','rb') as f:
	csfrac = pickle.load(f,encoding = 'latin1')

with open('isfraca.pkl','rb') as f:
	isfrac = pickle.load(f,encoding = 'latin1')
    
print(isfrac.shape)
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

slopes = np.zeros(len(w0array))
xdata = 10.0**10*epsilon
for i in range(0,len(w0array)):
    ydata = np.squeeze(isfrac[i,:,:])
    ydata = np.mean(ydata,1)
    ydata1 = np.squeeze(isfrac[i,:,:])
    ydata1 = np.mean(ydata1,1)
    a0 = np.max(ydata)
    b0 = 10
    x0a = np.arange(0.001,0.011,0.001)
    for j in range(len(x0a)):
        x0 = x0a[j]
        p0 =  [a0,0.1, b0, x0]
        p1 = [-0.1, 0.1]
        xmax = np.max(epsilon)
        xmin = np.min(epsilon)
        print(xmax, xmin)
        ymax = 1.0
        ymin = 0.0
        #popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox', bounds = ((0,-100,-20,0),(1,100,20,1.0)), maxfev=100000)
        popt, pcov = curve_fit(linearl, xdata, ydata,p1, method='dogbox')
        plt.figure(i)
        plt.plot(xdata, linearl(xdata, popt[0], popt[1]))
        #plt.plot(xdata, sigmoid(xdata, popt[0], popt[1], popt[2],popt[3]))
        slopes[i] = popt[0]
        #gdns[i,j] = pcov[0]
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"Fraction of time in state")
    plt.plot(xdata,ydata,'o')
    plt.show()


plt.plot(w0array, slopes)

plt.show()
"""
# fracm = np.squeeze(fracm)
plt.imshow(np.abs(sigmas), extent=[min(aa)-0.1,max(aa)+0.1,min(ga)-0.1,max(ga)+0.1], cmap = "turbo", aspect='auto',origin='lower')
#plt.clim(0,6)
plt.xlabel(r"average recurrent strength")
plt.ylabel(r"cross inhibition")
plt.title(r"$-k$")
plt.colorbar()
plt.show()


with open('fracttimecsc.pkl','rb') as f:
	fracm = pickle.load(f,encoding = 'latin1')

with open('fracttimeisc.pkl','rb') as f:
	fracmi = pickle.load(f,encoding = 'latin1')
sigmas = np.zeros((len(aa),len(ga)))
gdns = np.zeros((len(aa),len(ga)))

for i in range(0,len(aa)):
    for j in range(0,len(ga)):
        ydata = np.squeeze(fracm[i,:,j])
        ydata2 = np.squeeze(fracmi[i,:,j])
        #ydata = np.mean(ydata,1)
        print(ydata)
        p0 = [1.8]
        popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox')
        plt.plot(xdata,ydata,'o',color = "k")
        plt.plot(xdata, sigmoid(xdata,popt[0]),color = "k")
        sigmas[i,j] = popt[0]
        gdns[i,j] = pcov[0]
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"Fraction of time spent in CS")
plt.show()

# fracm = np.squeeze(fracm)
plt.imshow(np.abs(sigmas), extent=[min(aa)-0.1,max(aa)+0.1,min(ga)-0.1,max(ga)+0.1], cmap = "turbo", aspect='auto',origin='lower')
#plt.clim(0,6)
plt.xlabel(r"average recurrent strength")
plt.ylabel(r"cross inhibition")
plt.title(r"$-k$")
plt.colorbar()
plt.show()
"""