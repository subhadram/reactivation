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
from mpl_toolkits.mplot3d import Axes3D


dt = 1e-4
epsilon = np.arange(0.0e-9,1.1e-9, 0.1e-9)
#w0array = np.arange(0e-9,2.1e-9, 0.1e-9)
warray = np.arange(1.5e-9, 2.51e-9,1e-10)
#warray = np.arange(0,0.11 ,0.01)
#warray = np.arange(4, 41, 4)
#warray = np.logspace(-5,5, num =11)
print(len(warray), len(epsilon))

xdata = epsilon
csfrac = np.zeros((len(warray),len(epsilon),100))
isfrac = np.zeros((len(warray),len(epsilon),100))
print(csfrac.shape)
for tr in range(1,101):
    TR = str(tr)
    with open('csfraca6_{0}.pkl'.format(TR),'rb') as f: 
        csfrac[:,:,tr-1] = np.squeeze(pickle.load(f,encoding = 'latin1'))
    with open('isfraca6_{0}.pkl'.format(TR),'rb') as f:
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
#epsilon = np.arange(-0.2e-9, 0.21e-9, 1e-10)
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
    
 
#pickle.dump( alldatacs, open( "csfracillus.pkl","wb"))
#pickle.dump( alldatais, open( "isfracillus.pkl","wb"))
#import numpy, scipy.io
#scipy.io.savemat('csfracillus.mat', mdict={'csfracillus': alldatacs})

timea = np.arange(0,10000,0.1)
def expo(t,k,a0,ab):
    y = 25e-10*(0.1*np.exp(-k*t) + 0.3*np.exp(-k*t/8.0) )+ ab 
    return y

expon = expo(timea, 0.0005, 0.01, 1.5e-9)
expon2 = expo(timea-1800,0.0005, 0.1, 1.5e-9 )
plt.plot(timea, expon)
plt.plot(timea, expon2)
plt.show()

epsilon2 = np.arange(1.0e-9,0.0e-9, -0.01e-9)
warray2 = np.arange(2.5e-9,1.5e-9, -0.01e-9)
#warray2 = np.logspace(np.log(1.5e-9),np.log(2.5e-9), num =100)
#epsilon2 = np.logspace(np.log(0.1),np.log(1.1e-9),num=100)
plt.rcParams.update({'font.size': 14})
plt.imshow(alldatacs,origin='lower',extent = (min(epsilon), max(epsilon), min(warray), max(warray)),aspect = 'auto')
plt.plot(expon2-expon, expon, color = "red",zorder = 2, linewidth  = 2)

clb = plt.colorbar()
clb.ax.set_title('time in CS state',fontsize=12)
plt.xlabel(r"$\Delta$")
plt.ylabel("recurrent strength") 
plt.show()


plt.imshow(alldatais,extent = (min(epsilon), max(epsilon), max(warray), min(warray)),aspect = 'auto')
clb = plt.colorbar()
clb.ax.set_title('time in IS state above 10 Hz',fontsize=8)
plt.xlabel(r"$\Delta$")
plt.ylabel("recurrent strength") 
plt.show()


X2,Y2 = np.meshgrid(epsilon2, warray2)

X, Y = np.meshgrid(epsilon, warray)

X1 = X.ravel()
Y1 = Y.ravel()
Z = alldatacs


        

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y, Z, cmap='viridis')
ax.set_zlim(0,np.max(Z)+2)
plt.show()



# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
x, y = X.ravel(), Y.ravel()
z = alldatacs.ravel()


def sigmoidm(xy, L ,x0, k, b,c):
    x = xy[0]
    y = xy[1]
    z = y*(L / (1 + np.exp(-k*(x-x0)-c*(y))) + b) 
    return (z)

def min_function(params, x, y,z):
    model = sigmoidm(x,y, *params)
    residual = ((z - model) ** 2).sum()
    
    if np.any(model > 1):
        residual += 100  # Just some large value
    if np.any(model < 0):
        residual += 100

    return residual

from scipy.optimize import minimize, curve_fit
p2 = np.array([0,0,0,0,0])
result, pcov = curve_fit(sigmoidm, (x,y),z,p2, method='dogbox')
print(result)
plt.plot(sigmoidm((x,y), *result), label='model')
plt.plot(z)
plt.show()
print(*result)


from scipy.interpolate import Rbf,LinearNDInterpolator,griddata
rbf = Rbf(X, Y, Z, function="linear") 
#rbf = griddata((X, Y),Z,(epsilon2,warray2),method="linear" )
Z_pred = rbf(X, Y)

fig = plt.figure() 
ax = fig.add_subplot(projection="3d") 
ax.plot_surface(X, Y, Z) 
ax.plot_surface(X, Y, Z_pred) 
plt.show()
#warray2 = np.flip(np.logspace(1.5e-9,2.5e-9, num =100))
#epsilon2 = np.flip(np.logspace(0.0,1.1e-9,num=100))

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

plt.rcParams.update({'font.size': 14})
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
plt.plot(timea,trajc, color = "k",linewidth = 2, label = "CS only" )


print(len(epsilon2),len(warray2))
Z_traj= rbf(expon2-expon, expon)
plt.plot(timea, Z_traj,color = "red",linewidth = 2, label = "CS with IS")
plt.xlabel("time (s)")
plt.ylabel("time spent in CS state")
plt.legend()
plt.show()
"""
# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, fit, cmap='viridis')
cset = ax.contourf(X, Y, Z-fit, zdir='z', offset=-4, cmap='viridis')
ax.set_zlim(-4,np.max(fit))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(Z, origin='lower', cmap='viridis',
          extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(X, Y, fit, colors='w')
plt.show()
"""




