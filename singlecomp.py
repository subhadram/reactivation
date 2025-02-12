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
epsilon = np.arange(-0.3e-9,.31e-9, 0.1e-10)
w0array = np.arange(0e-9,2.1e-9, 0.2e-9)
warray = np.arange(2.1e-9, 2.21e-9,1.0e-10)
aarray = np.arange(1.5e-9,2.0e-9,.5e-9)
#epsilon = np.arange(0.0e-9, 0.525e-9, 2.5e-11)
#warray = np.arange(0,0.31 ,0.01)
#warray = np.arange(4, 41, 4)
#warray = np.logspace(-5,5, num =11)
print(warray)
trials = 100
xdata = epsilon
csfrac = np.zeros((len(aarray),len(epsilon),100))
isfrac = np.zeros((len(aarray),len(epsilon),100))
for tr in range(1,101):
    TR = str(tr)
    with open('csratead3_{0}.pkl'.format(TR),'rb') as f: 
        csfrac[:,:,tr-1] = np.squeeze(pickle.load(f,encoding = 'latin1'))
    with open('isratead3_{0}.pkl'.format(TR),'rb') as f:
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

slopes = np.zeros(len(warray))
slopes1 = np.zeros(len(warray))
xdata = 10.0**10*epsilon
timespent = np.zeros((len(warray),len(warray)))

plt.rcParams.update({'font.size': 15})
plt.errorbar(epsilon, np.mean(csfrac[1,:,:],1), yerr =np.std(csfrac[1,:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "forestgreen", label = "CS")
plt.errorbar(epsilon, np.mean(isfrac[1,:,:],1), yerr =np.std(isfrac[1,:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "gold", label = "IS")
plt.xlabel(r"$\Delta$")
plt.ylabel("fraction of time in state")
#plt.ylim(0,.8)
plt.legend()
#plt.ylim(0,0.08)
plt.show()



plt.imshow(timespent , extent=(min(warray), max(warray), min(slopes/slopes1), max(slopes/slopes1)), aspect = 'auto')
plt.plot(warray,slopes/slopes1,color="white",linewidth=4, zorder = 2)

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