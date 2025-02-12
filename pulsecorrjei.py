#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:00:46 2024

@author: subhadramokashe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:45:37 2023

@author: subhadramokashe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:07:43 2023

@author: subhadramokashe
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson

def main(trial):
    print(trial)
    N = 500
    G_L = 5e-9
    C = 100e-12
    E_L = -70e-3
    E_K = -80e-3
    V_Thresh = -50e-3
    V_Reset = -65e-3
    deltaT = 2e-3
    tauw = 100e-3   # Adaptation time constant (s)
    tauw2 = 1e-3
    taus = 20e-3
    tause = 20e-3
    tausi = 5e-3
    
    
    Vmax = 50e-3               # Level of voltage to detect and crop spikes
    
    
    dt = 1e-4           # dt in sec
    tmax = 20          # maximum time in sec
    tvector = np.arange(0,tmax,dt)
    
    # import scipy.io as sio
    # J = sio.loadmat('J.mat')
    # J = J['J']
    # print(J.shape)
    #plt.imshow(J)
    #plt.show()
    
    def spikes_to_rate(spikes, dt, window):
        windowsize = int(window/dt)
        sumspike = np.zeros_like(spikes)
        i = 0
        while i < len(spikes):
            sumspike[i:i+windowsize] = np.sum(spikes[i:i+windowsize])
            i = i + windowsize
            
        sumspike = sumspike/(window)
        return sumspike
            
    trials = 1
    
    aarray = np.arange(2.0e-9,2.01e-9,5e-11)
    #epsilon = np.arange(-0.1e-9, .11e-9, 1e-11)
    epsilon = np.arange(-0.30e-9,0.31e-9, 0.01e-9)
    #overlaps = np.arange(0.0, 0.06, 0.05)
    #warray = np.arange(0.0e-9, 11e-9, 1e-9)
    #firingrates = np.zeros((len(aarray),len(overlaps),trials,3,len(tvector)))
    csfrac = np.zeros((len(aarray),len(epsilon), trials))
    isfrac = np.zeros((len(aarray),len(epsilon), trials))
    csrate = np.zeros((len(aarray),len(epsilon), trials))
    israte = np.zeros((len(aarray),len(epsilon), trials))
    inhratel = np.zeros((len(aarray),len(epsilon), trials))
    inhrateh = np.zeros((len(aarray),len(epsilon), trials))
    mratecsd = np.zeros((len(aarray),len(epsilon), trials))
    mrateisd = np.zeros((len(aarray),len(epsilon), trials))
    mratecsb = np.zeros((len(aarray),len(epsilon), trials))
    mrateisb = np.zeros((len(aarray),len(epsilon), trials))
    mratecsa = np.zeros((len(aarray),len(epsilon), trials))
    mrateisa = np.zeros((len(aarray),len(epsilon), trials))
    neventis = np.zeros((len(aarray),len(epsilon), trials))
    neventcs = np.zeros((len(aarray),len(epsilon), trials))
    mratecstotal = np.zeros((len(aarray),len(epsilon), trials))
    mrateistotal = np.zeros((len(aarray),len(epsilon), trials))
    np.random.seed(int(trial))
    
        
    for k in range(len(aarray)):   
        
        for eps in range(len(epsilon)):
            
            for j in range(trials):
                a = aarray[k]
                TR = str(trial)
                #dw0 = warray[k]
                
                
               
                epsi = epsilon[eps]
                EP = str(epsi)
                J = np.zeros((N,N))
                Jee1 = a
                Jee2 = a + epsi
                Jei = 1.0e-9
                Jie = 0.8e-9
                overlap = 0.0
                group1 = 0.35
                group2 = 0.35
                dw0 = 1e-9
                dw2 = 1e-6
                ds = 1
                group1ei = int((group1)*N)
                group1ei = int((group1)*N)
                group2si = int((group1-overlap)*N)
                group2ei = int((group1+group2- overlap)*N)
                eind = np.zeros(N)
                eind[0:group2ei] = int(1)
                iind = np.zeros(N)
                iind[group2ei:] = int(1)
                inhind = int(0.75*N)
                
                J[inhind:,:inhind] = Jei
                J[:inhind,inhind:] = Jie
                J[:group1ei,:group1ei] = Jee1
                J[group2si:group2ei,group2si:group2ei ] =  Jee2
                dw = dw0*eind
                NE = inhind
                NI = N-NE
                Ng1 = int(0.35*N)
                Ng2 = int(0.35*N)
                Nt = len(tvector)
                #p = 0.1
                pii = 1.0
                pei = 0.25
                pie = 0.5
                pee = 0.1
                strc = np.block([[np.random.rand(inhind, inhind) < pee,np.random.rand(inhind, N - inhind) < pie],[np.random.rand(N - inhind, inhind) < pei,np.random.rand(N - inhind, N - inhind) < pii]])
                J = J * strc
                np.fill_diagonal(J, 0)
                #np.fill_diagonal(J, 0)
                #print(J[0,0])
                #plt.imshow(J)
                #plt.show()
                #lamda = 0.0
                #I = 15e-8*np.sqrt(dt)*(lamda*np.random.normal(0,1,[len(tvector)])+ np.sqrt(1- lamda**2)*np.random.normal(0,1,[N,len(tvector)]))
                #I[200:400,:] = 0.0
                #I[:,10000:] = 0.0
                V= np.zeros((N,len(tvector)))
                s= np.zeros((N,len(tvector)))
                w =  np.zeros((N,len(tvector)))
                w2 =  np.zeros((N,len(tvector)))
                spikes = np.zeros((N,len(tvector)))
                
                E_E = 0
                E_I = -80e-3
               
                # V00 = sio.loadmat('v00.mat')
                # V00 = V00['v00']
                # w00 = sio.loadmat('w00.mat')
                # w00 = w00['w00']
                # V[:,0] = np.squeeze(V00) #
                V[:,0] = E_L + (V_Thresh-E_L)*np.random.rand(N)
                #w[:,0] = np.squeeze(w00) #
                w[:,0] = dw*np.random.rand(N)
                #s[:,0] = ds*np.random.rand(N)
                inhindex = np.zeros(N)
                inhindex[0:inhind] = 1.0
                gnoise_E = 0.5e-9;
                gnoise_I = 0.5e-9;
                rnoise_EE = 3500;
                rnoise_IE = 1000;
                rnoise_EI = 1800;
                rnoise_II = 400;
                
                rnoise_EE_corr = 0;
                rnoise_IE_corr = 0;
                rnoise_EI_corr = 0;
                rnoise_II_corr = 0;
                
                tau_noiseE = 0.002;
                tau_noiseI = 0.005;
                """
                f = 1
                omega = 2*np.pi*f
                osc_rate = np.cos(omega*tvector)
                r_factor = 0.5*np.where(osc_rate +1 >= 0, osc_rate+1, 0)
                #r_factor = 0.5*np.max(osc_rate+1,0);
                
               """
                
               
                f = 4
                tau = 1 / f
                dtf = dt * f
                ran_r = np.random.rand(Nt)
                filt_rate = np.zeros_like(ran_r)
                filt_rate[0] = 0.5
                for i in range(1, Nt):
                    filt_rate[i] = filt_rate[i - 1] * (1 - dtf) + dtf * ran_r[i]
                filt_rate = filt_rate - np.mean(filt_rate)
                filt_rate = filt_rate / np.sqrt(dtf)
                r_factor1 = filt_rate#np.maximum(filt_rate + 0.5, 0)
                
                ran_r = np.random.rand(Nt)
                filt_rate = np.zeros_like(ran_r)
                filt_rate[0] = 0.5
                for i in range(1, Nt):
                    filt_rate[i] = filt_rate[i - 1] * (1 - dtf) + dtf * ran_r[i]
                filt_rate = filt_rate - np.mean(filt_rate)
                filt_rate = filt_rate / np.sqrt(dtf)
                r_factor2 = filt_rate#np.maximum(filt_rate + 0.5, 0)
                #plt.plot(r_factor)
                #plt.show()
                #r_factor = 0.8
                
                ran_r = np.random.rand(Nt)
                filt_rate = np.zeros_like(ran_r)
                filt_rate[0] = 0.5
                for i in range(1, Nt):
                    filt_rate[i] = filt_rate[i - 1] * (1 - dtf) + dtf * ran_r[i]
                filt_rate = filt_rate - np.mean(filt_rate)
                filt_rate = filt_rate / np.sqrt(dtf)
                r_factor3 = filt_rate#np.maximum(filt_rate + 0.5, 0)
                
                Nt250 = int(Nt/10000)
                
                r_factorg3 = np.zeros(Nt)
                
                for i in range(Nt250):
                    r_factorg3[10000*i +9500: 10000*(i+1) ] = 1.2
                    
                #plt.plot(tvector, r_factorg3)
                #plt.show()
                    
                
                alpha = np.sqrt(0.0)
                r_factorg1 = alpha*r_factor1 + np.sqrt(1-alpha*alpha)*r_factor3
                r_factorg1 = np.maximum(r_factorg1 + 0.5, 0)
                
                r_factorg2 = alpha*r_factor2 + np.sqrt(1-alpha*alpha)*r_factor3
                r_factorg2 = np.maximum(r_factorg2 + 0.5, 0)
                
                
                #r_factorg3 = np.maximum(r_factor3 + 0.5, 0)
    
                
    
                noise_EE = gnoise_E * (poisson.rvs(rnoise_EE * dt, size=(NE, Nt)) +
                                        np.outer(np.ones(NE), poisson.rvs(rnoise_EE_corr * dt, size=Nt)))
                noise_EE[0:group1ei,:] = noise_EE[0:group1ei,:] * (np.random.rand(Ng1, Nt) < np.outer(np.ones(Ng1), (r_factorg3)))
                noise_EE[group2si:group2ei,:] = noise_EE[group2si:group2ei,:] * (np.random.rand(Ng2, Nt) < np.outer(np.ones(Ng2), (r_factorg3)))
                noise_EE[group2ei:inhind,:] = noise_EE[group2ei:inhind,:] * (np.random.rand(int(0.05*N), Nt) < np.outer(np.ones(int(0.05*N)), r_factorg3))
    
                noise_IE = gnoise_I * (poisson.rvs(rnoise_IE * dt, size=(NE, Nt)) +
                                        np.outer(np.ones(NE), poisson.rvs(rnoise_IE_corr * dt, size=Nt)))
                noise_EI = gnoise_E * (poisson.rvs(rnoise_EI * dt, size=(NI, Nt)) +
                                        np.outer(np.ones(NI), poisson.rvs(rnoise_EI_corr * dt, size=Nt)))
                noise_II = gnoise_I * (poisson.rvs(rnoise_II * dt, size=(NI, Nt)) +
                                        np.outer(np.ones(NI), poisson.rvs(rnoise_II_corr * dt, size=Nt)))
    
                gnoise_EE = np.zeros((NE, Nt))
                gnoise_IE = np.zeros((NE, Nt))
                gnoise_EI = np.zeros((NI, Nt))
                gnoise_II = np.zeros((NI, Nt))
        
                for i in range(1, len(tvector)):
                    gnoise_EE[:, i - 1] = gnoise_EE[:, i - 1] + noise_EE[:, i - 1]
                    gnoise_IE[:, i - 1] = gnoise_IE[:, i - 1] + noise_IE[:, i - 1]
                    gnoise_EI[:, i - 1] = gnoise_EI[:, i - 1] + noise_EI[:, i - 1]
                    gnoise_II[:, i - 1] = gnoise_II[:, i - 1] + noise_II[:, i - 1]
        
                    gnoise_EE[:, i] = gnoise_EE[:, i - 1] * np.exp(-dt / tau_noiseE)
                    gnoise_IE[:, i] = gnoise_IE[:, i - 1] * np.exp(-dt / tau_noiseI)
                    gnoise_EI[:, i] = gnoise_EI[:, i - 1] * np.exp(-dt / tau_noiseE)
    
                for tt in range(len(tvector)-1):
                    cellsfired = np.argwhere(V[:,tt] >= Vmax)
                    cellsfirede = cellsfired[cellsfired< group2ei]
                    cellsfiredi = cellsfired[cellsfired>= group2ei]
                    #print(cellsfirede)
                    V[cellsfired,tt] = V_Reset
                    w[cellsfired,tt] = w[cellsfired,tt] + dw[cellsfired]
                    w2[cellsfired,tt] = w2[cellsfired,tt] + dw2
                    s[cellsfired,tt] = s[cellsfired,tt] + ds*(1.0-s[cellsfired,tt])
                    spikes[cellsfired,tt] =  1
                   # print(gnoise_EE[:,tt].shape, np.matmul(J[:,:inhind],s[:inhind,tt]).shape)
                    inpe = np.zeros(N)
                    inpi = np.zeros(N)
                    inpe[:NE] = gnoise_EE[:,tt]
                    inpe[NE:] = gnoise_EI[:,tt]
                    inpi[:NE] = gnoise_IE[:,tt]
                    inpi[NE:] = gnoise_II[:,tt]
                    #inputsfrome = np.matmul(J[:,:inhind],s[:inhind,tt]) + inpe
                    #inputsfromi = np.matmul(J[:,inhind:],s[inhind:,tt]) + inpi
                    inputsfrome = np.dot(J[:, :inhind], s[:inhind, tt]) + np.concatenate([gnoise_EE[:, tt], gnoise_EI[:, tt]])
                    inputsfromi = np.dot(J[:, inhind:], s[inhind:, tt]) + np.concatenate([gnoise_IE[:, tt], gnoise_II[:, tt]])
                    vn = V[:,tt]
                    dv1 = dt*( G_L*(E_L-vn + deltaT*np.exp((vn-V_Thresh)/deltaT) )+ (w2[:,tt]+w[:,tt])*(E_K-vn) + inputsfrome*(E_E - vn) + inputsfromi*(E_I - vn))/C
                    vn = V[:,tt] + 0.5*dv1
                    dv2 = dt*( G_L*(E_L-vn + deltaT*np.exp((vn-V_Thresh)/deltaT) )+ (w2[:,tt]+w[:,tt])*(E_K-vn) + inputsfrome*(E_E - vn) + inputsfromi*(E_I - vn))/C
                    vn = V[:,tt] + 0.5*dv2
                    dv3 = dt*( G_L*(E_L-vn + deltaT*np.exp((vn-V_Thresh)/deltaT) )+ (w2[:,tt]+w[:,tt])*(E_K-vn) + inputsfrome*(E_E - vn) + inputsfromi*(E_I - vn))/C
                    vn = V[:,tt] + dv3
                    dv4 = dt*( G_L*(E_L-vn + deltaT*np.exp((vn-V_Thresh)/deltaT) )+ (w2[:,tt]+w[:,tt])*(E_K-vn) + inputsfrome*(E_E - vn) + inputsfromi*(E_I - vn))/C
                    V[:,tt + 1 ] = V[:,tt] + (dv1+2*dv2+2*dv3+dv4)/6.0
                    V[:, tt+1 ] = np.where(V[:, tt+1] >= E_K, V[:, tt+1], E_K )
                    V[:, tt+1 ] = np.where(V[:, tt+1] <=  (Vmax+1e-3), V[:, tt+1], (Vmax+1e-3 ))
                    
                    s[:inhind,tt+1] = s[:inhind,tt]*np.exp(-dt/tause)
                    s[inhind:, tt+1 ] =  s[inhind:,tt]*np.exp(-dt/tausi)
                    
                    
                    w[:,tt+1] = w[:,tt]*np.exp(-dt/tauw)
                    w2[:,tt+1] = w2[:,tt]*np.exp(-dt/tauw2)
                #plt.plot(tvector,V[0,:])
                #plt.plot(tvector,V[400,:])
                #plt.show()
                spikesg1 = np.mean(spikes[:group2si,:],0)
                spikesg2 = np.mean(spikes[group2si + round(overlap * N) + 1:group2ei, :],0)
                spikesov = np.mean(spikes[inhind + 1:,:],0)
                #firingrates[k,i,j,0,:] = spikesg1
                #firingrates[k,i,j,1,:] = spikesg2
                #firingrates[k,i,j,2,:] = spikesov
                rateg1 = spikes_to_rate(spikesg1, dt, 0.02)
                rateg2 = spikes_to_rate(spikesg2, dt, 0.02)
                rateov = spikes_to_rate(spikesov, dt, 0.02)
                # fig = plt.figure(1)
                # plt.plot(tvector, rateg1)
                # plt.plot(tvector, rateg2)
                # #plt.plot(tvector, rateov)
                # plt.title(r" $\epsilon$ = "+ EP)
                # plt.savefig('pfrate_%s_%s.png'%(EP,TR))
                # # #plt.show()
                # plt.close(fig)
                # fig = plt.figure(2)
                # plt.hist(rateov)
                # plt.xlabel("firing rate")
                # plt.savefig('hist_%s_%s.png'%(EP,TR))
                # plt.show()
                # plt.close(fig)
                # maxratecs[i,j] =  np.max(rateg1[1000:])
                # maxrateis[i,j] =  np.max(rateg2[1000:])
                # b = (rateg2-rateg1)
                # fractinis = np.where(b> 2.0, 1,0 )
                # fractincs = np.where(b< -2.0, 1,0 )
                # spikescs[i,j] = np.sum(spikes[:group2si,1000:])
                # spikesis[i,j] =  np.sum(spikes[group2si+int(overlap*N):group2ei,1000:])
                thigh1 = np.mean(rateg1 > 10)
                thigh2 = np.mean(rateg2 > 10)
                #print([i for i,x in enumerate(rateov) if rateg1[i] <= 10 and rateg2[i]<=10])
                thighli = [i for i,x in enumerate(rateov) if rateg1[i] <= 10 and rateg2[i]<=10]
                thighhi = [i for i,x in enumerate(rateov) if rateg1[i] > 10 or rateg2[i]>10]
              
                #mratecstotal = np.mean(rateg1)
                #mrateistotal = np.mean(rateg2)
                
                thighl = rateov[thighli]
                thighh = rateov[thighhi]
                inpts = rateg1>10 
                #print(inpts)
                timevec = tvector[1:]
                prod1 = inpts[1:]!=inpts[:-1]
                prod1 = np.where(prod1==1)
                #print(prod1)
                prod11 = timevec[inpts[1:]!=inpts[:-1]]
                prod12 = np.diff(prod11)
                #print(prod12)
                #plt.plot(prod11)
                #plt.show()
                #print(prod12)
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
                    
                during1 = []
                after1 = []
                before1= []
                for ie,ei in enumerate(event_indices):
                    during1.append(np.mean(rateov[ei:ei+500]))
                    after1.append(np.mean(rateov[ei+500:ei+1000]))
                    before1.append(np.mean(rateov[ei-500:ei]))
                    
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
                
                during2 = []
                after2 = []
                before2 = []
                for ie,ei2 in enumerate(event_indices2):
                    during2.append(np.mean(rateov[ei2:ei2+500]))
                    #during2.append(np.mean(rateov[ei2:ei2+500]))
                    after2.append(np.mean(rateov[ei2+500:ei2+1000]))
                    before2.append(np.mean(rateov[ei2-500:ei2]))
                    
                # if len(during2)==0:
                #     during2 = [0]
                # if len(during1)==0:
                #     during1 = [0]
                # if len(after1)==0:
                #     after1 = [0]
                # if len(after2)==0:
                #     after2 = [0]
                # if len(before1)==0:
                #     before1 = [0]
                # if len(before2)==0:
                #     before2 = [0]
                
                csfrac[k,eps, j] = thigh1
                isfrac[k,eps, j] = thigh2
                csrate[k,eps, j] = prod13
                israte[k,eps, j] = prod23
                inhratel[k,eps, j] = np.mean(thighl)
                inhrateh[k,eps,j] = np.mean(thighh)
                mratecsd[k,eps,j] = np.mean(during1)
                mrateisd[k,eps,j] = np.mean(during2)
                mratecsb[k,eps,j] = np.mean(before1)
                mrateisb[k,eps,j] = np.mean(before2)
                mratecsa[k,eps,j] = np.mean(after1)
                mrateisa[k,eps,j] = np.mean(after2)
                neventcs[k,eps,j] = lencs
                neventis[k,eps,j] = lenis
                mratecstotal[k,eps,j] = np.mean(rateg1)
                mrateistotal[k,eps,j] = np.mean(rateg2)
                
                print(k,eps,j)
                #print(np.mean(thighl),np.mean(thighh))
            
                
                #plt.title(r"g =" + GG + r" $\epsilon$ = "+ EP + " a =" + AA + r" $\sigma_x$ = " + NAMP + " tr = " + TR)
               # plt.savefig('xx_g_%s_ep_%s_a_%s_ic_%s_na_%s_tr_%s.png'%(GG, EP,AA,LL,NAMP,TR))
               # plt.close(fig)
    #print(csrate)
    #print(israte)
    # plt.plot(np.squeeze(mratecsd))
    # plt.plot(np.squeeze(mrateisd))
    # plt.show()
    
    # plt.plot(np.squeeze(mratecsb))
    # plt.plot(np.squeeze(mrateisb))
    # plt.show()
    
    # plt.plot(np.squeeze(mratecsa))
    # plt.plot(np.squeeze(mrateisa))
    # plt.show()
    
    # plt.plot(np.squeeze(csrate)*np.squeeze(neventcs),color = "forestgreen")
    # plt.plot(np.squeeze(israte)*np.squeeze(neventis),color = "gold")
    # plt.plot(np.squeeze(csfrac)*tmax, "o", color = "forestgreen")
    # plt.plot(np.squeeze(isfrac)*tmax, "o", color = "gold")
    # plt.show()
    print(mratecsd)
    print(mrateisd)
    
    #pickle.dump(mratecsd,open('u2dratecs_{0}.pkl'.format(TR),'wb'))
    #pickle.dump(mrateisd,open('u2drateis_{0}.pkl'.format(TR),'wb'))
    #pickle.dump(mratecsb,open('u2bratecs_{0}.pkl'.format(TR),'wb'))
    #pickle.dump(mrateisb,open('u2brateis_{0}.pkl'.format(TR),'wb'))
    #pickle.dump(mratecsa,open('u2aratecs_{0}.pkl'.format(TR),'wb'))
    #pickle.dump(mrateisa,open('u2arateis_{0}.pkl'.format(TR),'wb'))
    pickle.dump(csfrac,open('pulsejeicsfrac_{0}.pkl'.format(TR),'wb'))
    pickle.dump(isfrac,open('pulsejeiisfrac_{0}.pkl'.format(TR),'wb'))
    #pickle.dump(lencs,open('u2numcs_{0}.pkl'.format(TR),'wb'))
    #pickle.dump(lenis,open('u2numis_{0}.pkl'.format(TR),'wb'))
    pickle.dump(csrate,open('pulsejeicsrate{0}.pkl'.format(TR),'wb'))
    pickle.dump(israte,open('pulsejeiisrate_{0}.pkl'.format(TR),'wb'))
    pickle.dump(mratecstotal,open('pulsejeiratecst_{0}.pkl'.format(TR),'wb'))
    pickle.dump(mrateistotal,open('pulsejeirateist_{0}.pkl'.format(TR),'wb'))
    
    #pickle.dump(inhratel,open('inhratelinh_{0}.pkl'.format(TR),'wb'))
    #pickle.dump(inhrateh,open('inhratelinh_{0}.pkl'.format(TR),'wb'))
    
#plt.plot(epsilon, np.mean(csfrac,2))
#plt.plot(epsilon, np.mean(isfrac,2))
#plt.show()
import sys
if __name__ == '__main__':
    args = sys.argv
    print(args[1])
    main(args[1])
