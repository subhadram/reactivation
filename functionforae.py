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
epsilon = np.arange(0.0e-9,1.1e-9, 0.1e-9)
w0array = np.arange(0e-9,2.1e-9, 0.2e-9)
warray = np.arange(1.5e-9, 2.51e-9,0.1e-10)
#warray = np.arange(0,0.11 ,0.01)
#warray = np.arange(4, 41, 4)
#warray = np.logspace(-5,5, num =11)
print(warray)

xdata = epsilon
csfrac = np.zeros((len(warray),len(epsilon),1000))
isfrac = np.zeros((len(warray),len(epsilon),1000))
for tr in range(1,1001):
    TR = str(tr)
    with open('csfraca5_{0}.pkl'.format(TR),'rb') as f: 
        csfrac[:,:,tr-1] = np.squeeze(pickle.load(f,encoding = 'latin1'))
    with open('isfraca5_{0}.pkl'.format(TR),'rb') as f:
        isfrac[:,:,tr-1] = np.squeeze(pickle.load(f,encoding = 'latin1'))
        

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
epsilon = np.arange(-0.2e-9, 0.21e-9, 1e-10)
slopes = np.zeros(len(warray))
slopes1 = np.zeros(len(warray))
xdata = 10.0**10*epsilon
timespent = np.zeros((len(warray),len(warray)))
alldatacs = np.zeros((len(warray), len(epsilon)))
alldatais = np.zeros((len(warray), len(epsilon)))
for i in range(0,len(warray)):
    ydata = np.squeeze(csfrac[i,:,:])
    ydata = np.mean(ydata,1)
    ydata1 = np.squeeze(isfrac[i,:,:])
    ydata1 = np.mean(ydata1,1)
    alldatacs[i,:] = ydata
    alldatais[i,:] = ydata1
    a0 = np.max(ydata)
    b0 = 10
    x0a = np.arange(0.01,0.11,0.01)

    x0 = 0.01
    p0 =  [a0,0.1, b0, x0]
    p1 = [-0.1, 0.01]
    p2 = [-0.1, 0.01]
    xmax = np.max(epsilon)
    xmin = np.min(epsilon)
    print(xmax, xmin)
    ymax = 1.0
    ymin = 0.0
    #popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox', bounds = ((0,-100,-20,0),(1,100,20,1.0)), maxfev=100000)
    popt, pcov = curve_fit(linearl, xdata, ydata,p1, method='dogbox')
    popt1, pcov1 = curve_fit(linearl, xdata, ydata1,p2, method='dogbox')
    #plt.figure(i)
    plt.plot(xdata, linearl(xdata, popt[0], popt[1]))
    plt.plot(xdata, linearl(xdata, popt1[0], popt1[1]))
    slopes[i] = popt[0]
    slopes1[i] = popt1[0]
    timespent[:,i] = ydata[3]
    #gdns[i,j] = pcov[0]
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"Fraction of time in state")
    plt.plot(xdata,ydata,'o')
    plt.plot(xdata,ydata1,'o')
plt.show()

plt.imshow(alldatacs,extent = (min(epsilon), max(epsilon), max(warray), min(warray)),aspect = 'auto')
clb = plt.colorbar()
clb.ax.set_title('time in CS state above 10 Hz',fontsize=8)
plt.xlabel(r"$\epsilon$")
plt.ylabel("a") 
plt.show()
plt.imshow(alldatais,extent = (min(epsilon), max(epsilon), max(warray), min(warray)),aspect = 'auto')
clb = plt.colorbar()
clb.ax.set_title('time in IS state above 10 Hz',fontsize=8)
plt.xlabel(r"$\epsilon$")
plt.ylabel("a") 
plt.show()

plt.imshow(timespent , extent=(min(warray), max(warray), min(np.abs(slopes/slopes1)), max(np.abs(slopes/slopes1))), aspect = 'auto')
plt.plot(warray,np.abs(slopes/slopes1),color="white",linewidth=4, zorder = 2)

#ax2.set_xticks([])
#ax2.set_yticks([])
#plt.plot(warray, slopes1, label ="IS")
#plt.xscale('log')
#plt.xlim(0,0.3)
plt.ylabel("ratio of slopes")
plt.xlabel("overlap")

clb = plt.colorbar()
clb.ax.set_title('average rate',fontsize=8)
plt.show()

from scipy.interpolate import Rbf 

X, Y = np.meshgrid(epsilon, warray)

rbf = Rbf(X, Y, alldatais, function="linear")
epsilon2 =  np.arange(-1.2e-9, 1.21e-9, 1e-10)
X2, Y2 = np.meshgrid(epsilon2, warray)
Z_pred = rbf(X, Y)

fig = plt.figure() 
ax = fig.add_subplot(projection="3d") 
ax.plot_surface(X, Y, alldatais) 
ax.plot_surface(X, Y, Z_pred) 
plt.show()



