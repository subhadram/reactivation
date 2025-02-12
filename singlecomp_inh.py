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
epsilon = np.arange(-0.30e-9,0.31e-9, 0.01e-9)
epsilon = np.arange(-0.3e-9, .31e-9, 1e-11)
#epsilon = np.arange(-0.1e-9, .11e-9, 1e-11)
w0array = np.arange(0e-9,2.1e-9, 0.2e-9)
warray = np.arange(2.1e-9, 2.1e-9,1.0e-10)
aarray = np.arange(1.5e-9,1.51e-9,.5e-9)
#epsilon = np.arange(0.0e-9, 0.525e-9, 2.5e-11)
#warray = np.arange(0,0.31 ,0.01)
#warray = np.arange(4, 41, 4)
#warray = np.logspace(-5,5, num =11)
print(warray)
trials = 100
xdata = epsilon
csfrac = np.zeros((len(aarray),len(epsilon),trials))
isfrac = np.zeros((len(aarray),len(epsilon),trials))
for tr in range(1,101):
    TR = str(tr)
    with open('pf05mcsfrac_{0}.pkl'.format(TR),'rb') as f: 
        csfrac[:,:,tr-1] = np.squeeze(pickle.load(f,encoding = 'latin1'))
    with open('pf05misfrac_{0}.pkl'.format(TR),'rb') as f:
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
print(csfrac)
plt.rcParams.update({'font.size': 15})
plt.errorbar(epsilon, np.nanmean(csfrac[0,:,:],1), yerr =np.nanstd(csfrac[0,:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "forestgreen", label = "CS")
plt.errorbar(epsilon, np.nanmean(isfrac[0,:,:],1), yerr =np.nanstd(isfrac[0,:,:],1)/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "gold", label = "IS")
plt.xlabel(r"$\Delta$")
plt.ylabel("fraction of time spent")
#plt.ylabel("interneurons firing frequency (Hz)")
#plt.ylim(0,.8)
plt.legend()
#plt.ylim(0,0.08)
plt.show()

p1 = [-0.1, 0.01]
p2 = [-0.1, 0.01]
xdata = 10.0**10*epsilon
ydata = np.squeeze(np.nanmean(csfrac[0,:,:],1))
ydata1= np.squeeze(np.nanmean(isfrac[0,:,:],1))
popt, pcov = curve_fit(linearl, xdata, ydata,p1, method='dogbox')
popt1, pcov1 = curve_fit(linearl, xdata, ydata1,p2, method='dogbox')
#plt.figure(i)
plt.plot(xdata, linearl(xdata, popt[0], popt[1]))
plt.plot(xdata, linearl(xdata, popt1[0], popt1[1]))
slopes= popt[0]
slopes1 = popt1[0]
print(slopes/slopes1)
#plt.imshow(timespent , extent=(min(warray), max(warray), min(slopes/slopes1), max(slopes/slopes1)), aspect = 'auto')
#plt.plot(warray,slopes/slopes1,color="white",linewidth=4, zorder = 2)

#ax2.set_xticks([])
#ax2.set_yticks([])
#plt.plot(warray, slopes1, label ="IS")
#plt.xscale('log')
#plt.xlim(0,0.3)
#plt.ylabel("ratio of slopes")
#plt.xlabel("overlap")

#clb = plt.colorbar()
#clb.ax.set_title('average rate',fontsize=8)
#plt.show()
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