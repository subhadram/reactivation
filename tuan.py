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



def linearl(x, m, c):
    y = m*x+c
    return y


trials =200
mus = np.arange(1.5,2.51,0.1) ############################# mu of the transfer function####
sigma = 0.1 ######################### sigma of the transfer function###
tau = 5
namp= 1.0
lambda_i = 0.5
epsilon = np.arange(-0.30,0.31,0.06)
ga = np.arange(1.5, 2.51, 0.1)
aa = np.arange(4.0,4.1,0.2)
xdata = epsilon
csfrac = np.zeros((len(ga),len(epsilon),200))
isfrac = np.zeros((len(ga),len(epsilon),200))

csfrac = np.zeros((len(ga),len(epsilon),len(aa),1000))
isfrac = np.zeros((len(ga),len(epsilon),len(aa),1000))
uptimecs = np.zeros((len(ga),len(epsilon),len(aa),1))
uptimeis = np.zeros((len(ga),len(epsilon),len(aa),1))
#x = np.zeros((len(ga),len(epsilon),len(aa),len(mus),na,len(timea)))
#phix = np.zeros((len(ga),len(epsilon),len(aa),len(mus),na,len(timea)))
trans_to_csstate = np.zeros((len(ga),len(epsilon),len(aa),len(mus)))
start_state_ll= np.zeros((len(ga),len(epsilon),len(aa),len(mus)))
time_to_first_transition = np.zeros((len(ga),len(epsilon),len(aa),len(mus)))
trials = 1
for tr in range(1,1001):
    TR = str(tr)
    with open('fracttimecsc0_{0}.pkl'.format(TR),'rb') as f: 
        csfrac0[:,:,:,tr-1] = pickle.load(f,encoding = 'latin1')
    with open('fracttimeisc0_{0}.pkl'.format(TR),'rb') as f:
        isfrac0[:,:,:,tr-1] = pickle.load(f,encoding = 'latin1')
        
for tr in range(1,1001):
    TR = str(tr)
    with open('fracttimecsc25_{0}.pkl'.format(TR),'rb') as f: 
        csfrac[:,:,:,tr-1] = pickle.load(f,encoding = 'latin1')
    with open('fracttimeisc25_{0}.pkl'.format(TR),'rb') as f:
        isfrac[:,:,:,tr-1] = pickle.load(f,encoding = 'latin1')


# for j in range(len(ga)):
#     plt.plot(np.squeeze(np.mean(csfrac[j,:,:,:],2)))
#     plt.plot(np.squeeze(np.mean(isfrac[j,:,:,:],2)))
    
#     plt.show()
#     #with open('isfraca.pkl','rb') as f:
# 	#isfrac = pickle.load(f,encoding = 'latin1')
    


# def sigmoid(x, a,b,c):
#     y = a / (1 + np.exp(-b*(x))) + c
#     return (y)
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)
def linearl(x, m, c):
    y = m*x+c
    return y

slopes = np.zeros(len(ga))
slopes1 = np.zeros(len(ga))
#xdata = 10.0**10*epsilon
#timespent = np.zeros((len(warray),len(warray)))

plt.rcParams.update({'font.size': 15})
plt.errorbar(epsilon, np.squeeze(np.mean(csfrac[-1,:,:,:],2)), yerr =np.squeeze(np.std(csfrac[0,:,:,:],2))/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "forestgreen", label = "CS")
plt.errorbar(epsilon, np.squeeze(np.mean(isfrac[-1,:,:,:],2)), yerr =np.squeeze(np.std(isfrac[0,:,:,:],2))/np.sqrt(trials),solid_capstyle='projecting', capsize=5,color = "gold", label = "IS")
plt.xlabel(r"$\Delta$")
plt.ylabel("fraction of time in state")
#plt.ylim(0,.8)
plt.legend()
#plt.ylim(0,0.08)
plt.show()


slopes = np.zeros(len(ga))
slopes1 = np.zeros(len(ga))



for i in range(len(ga)):
    xdata = epsilon
    ydata = np.squeeze(np.mean(csfrac[i,:,:,:],2))
    ydata1 = np.squeeze(np.mean(isfrac[i,:,:,:],2))
    p1 = [-0.1, 0]
    p2 = [0.1, 0]
    xmax = np.max(epsilon)
    xmin = np.min(epsilon)
    
    ymax = 1.0
    ymin = 0.0
    #popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox', bounds = ((0,-100,-20,0),(1,100,20,1.0)), maxfev=100000)
    popt, pcov = curve_fit(linearl, xdata, ydata,p1, method='dogbox')
    popt1, pcov1 = curve_fit(linearl, xdata, ydata1,p2, method='dogbox')
    #plt.figure(i)
    #plt.plot(xdata, linearl(xdata, popt[0], popt[1]))
    #plt.plot(xdata, linearl(xdata, popt1[0], popt1[1]))
    
    slopes[i] = popt[0]
    slopes1[i] = popt1[0]
    print(slopes[i], slopes1[i])
    
    


plt.plot(ga, slopes/slopes1,)
plt.xlabel(r"$\mu$")
plt.ylabel("ratio of slopes")
plt.show()
    
    