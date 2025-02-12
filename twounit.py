
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib
import pickle


def phi(xx,mu,sig):
		fy = 1/2.0*(1+erf((xx-mu)/(np.sqrt(2)*sig)))
		return fy

def phi2(xx,mu,sig,namp):
		fy = 1/2.0*(1+erf((xx-mu)/(np.sqrt(2*sig**2 + 2*namp**2))))
		return fy
    
def main(trial):
    na = 2
    
    trials = 1
    dt = 1.0
    tend = 100000
    timea = np.arange(0.0,tend, dt)
    x = np.zeros((trials,na,len(timea)))
    #mu = 2.0 ############################# mu of the transfer function####
    sigma = 0.1 ######################### sigma of the transfer function###
    tau = 5
    namp= 1.0
    lambda_i = 0.5
    epsilon = np.arange(-0.30,0.31,0.06)
    mus = np.arange(1.5,2.51,0.1)
    aa = np.arange(4.0,4.1,0.2)
    print(mus)
    I = 0*np.random.normal(0,1,[na, len(timea)])
    TR = str(trial)
    xi = [0.0101,0.01]
    xi = np.array(xi)
    xlast1 = np.zeros((len(mus),len(epsilon),len(aa)))
    xlast2 = np.zeros((len(mus),len(epsilon),len(aa)))
    
    fractimeis = np.zeros((len(mus),len(epsilon),len(aa)))
    fractimecs = np.zeros((len(mus),len(epsilon),len(aa)))
    uptimecs = np.zeros((len(mus),len(epsilon),len(aa)))
    uptimeis = np.zeros((len(mus),len(epsilon),len(aa)))
    x = np.zeros((len(mus),len(epsilon),len(aa),trials,na,len(timea)))
    phix = np.zeros((len(mus),len(epsilon),len(aa),trials,na,len(timea)))
    trans_to_csstate = np.zeros((len(mus),len(epsilon),len(aa),trials))
    start_state_ll= np.zeros((len(mus),len(epsilon),len(aa),trials))
    time_to_first_transition = np.zeros((len(mus),len(epsilon),len(aa),trials))
    for k in range(0,len(mus)):
        
        mu = mus[k]
        for j in range(0, len(epsilon)):
          epsi = epsilon[j]
          #print(J)
          fig = plt.figure(1)
          GG= str(mu)
          EP = str(epsi)  
          g = 3.0
          
          NAMP= str(namp)
          for l in range(len(aa)):
            LL = str(l)
            a = aa[l]
            AA = str(a)
            J = [[a, -1.0*g],[-1.0*g,a+epsi]]
            fractimetemp = np.zeros(trials)
            fractimetempcs = np.zeros(trials)
            fractimetempis = np.zeros(trials)
           
            for h in range(0,trials):
                x[k,j,l,h,0,0] = xi[0]
                x[k,j,l,h,1,0] = xi[1]
                #np.random.seed(9)
                Ix = np.random.normal(0,1,[na, len(timea)]) ####noise input
                Ix[1,:] = Ix[1,:]*np.sqrt(1 - lambda_i*lambda_i) + lambda_i*Ix[0,:]
                Ix = namp*Ix
                #TR = str(h)
                for i in range(0, len(timea)-1):
                    x[k,j,l,h,:,i + 1] = x[k,j,l,h,:,i] + (-x[k,j,l,h,:,i] + phi(np.dot(J,x[k,j,l,h,:,i]) + Ix[:,i], mu, sigma))*dt/tau
                    phix[k,j,l,h,:,i + 1] = phi(np.dot(J,x[k,j,l,h,:,i]) + Ix[:,i], mu, sigma)
                         
                xup = x[k,j,l,h,1,:]
                xd = x[k,j,l,h,0,:]
                xin = xd-xup
            
                xin = np.squeeze(xin)
                #print(xin.shape)
                xin2 = xin.tolist()
                xdl = np.squeeze(xd)
                xdl = xdl.tolist()
                xupl = np.squeeze(xup)
                xupl = xupl.tolist()
                #res = [idx for idx in range(0, len(xin2) - 1) if xin2[idx] >0.0 and xin2[idx + 1] < 0.0]
                
                start_state_ll[k,j,l,h] = np.where(np.abs(xupl[5])<0.5 and np.abs(xdl[5])<0.5,1,0)
                #print(start_state_ll[k,j,l,h], "state")
                
                t = 5
                while t < tend- 12:
                    if (xin2[t+5] >= 0) and (xdl[t+5] - xupl[t-5] > 0.3) and (xdl[t+10] > 0.75) :
                        trans_to_csstate[k,j,l,h] = 1.0
                        time_to_first_transition[k,j,l,h] = t*dt
                        t = tend
                    elif (xin2[t+5] <= 0) and (-xdl[t-5] + xupl [t+5] > 0.3) and (xupl[t+10] > 0.75):
                        time_to_first_transition[k,j,l,h] = t*dt
                        t = tend
                    t =  t + 2
                        
                        
                    #res = [idx for idx in range(5, len(xin2) - 20,6) if (xin2[idx-5] < 0.3) and (xin2[idx+5] >= 0.3) and (xdl[idx+10] > 0.75)]
                    #resex = [idx for idx in range(5, len(xin2) - 20,6) if (xin2[idx-5] > 0.3) and (xin2[idx+5] <= 0.3) and (xdl[idx+10] < 0.75)]
                    #print(res)
        
                    
                
                #frcs = np.array(np.where(xd-xup >= 0.7,1,0))
                #fris = np.array(np.where(xup-xd >= 0.7,1,0))
                #print(sum(frcs),sum(fris) )
                #res = [idx for idx in range(0, len(fris) - 1) if fris[idx] >0 and fris[idx + 1] <= 0]
                frcs = np.array(np.where(xd >= 0.8,1,0))
                fris = np.array(np.where(xup >= 0.8,1,0))
                #print(frcs)
                fractimetempis[h] = np.mean(fris)
                fractimetempcs[h] = np.mean(frcs)
                
                
                #print(fris[1000:1002],xup[1000:1002] )
                #resiso = [idx for idx in range(5, len(xup) - 5) if xup[idx] < 0.8 and np.mean(xup[idx - 5:idx]) >= 0.8 and np.mean(xup[idx:idx+5]) < 0.6]
                #resisi = [idx for idx in range(5, len(xup) - 5) if xup[idx] >= 0.8 and np.mean(xup[idx - 5:idx]) < 0.6 and np.mean(xup[idx:idx + 5])>= 0.8]
                #isit = [resiso[idx]- resisi[idx] for idx in range(0, min(len(resiso),len(resisi))-5) if((resiso[idx]> resisi[idx]+10))]
                # rescso = [idx for idx in range(0, len(frcs) - 1) if frcs[idx] ==1 and frcs[idx + 1] == 0]
                # rescsi = [idx for idx in range(0, len(frcs) - 1) if frcs[idx] ==0 and frcs[idx + 1] == 1]
                # csit = [rescso[idx]- rescsi[idx] for idx in range(0, min(len(rescso),len(rescsi))-5) if((rescso[idx]> rescsi[idx]+10) and (xd[rescso[idx]+2]<0.6))]
                fig = plt.figure(1)
                plt.plot(timea, x[k,j,l,h,0,:], label = "CS")
                plt.plot(timea, x[k,j,l,h,1,:],label = "IS")
                plt.plot(time_to_first_transition[k,j,l,h], trans_to_csstate[k,j,l,h] , "o", label="first transtion")
                #plt.plot(timea,fris,'o')
                #plt.plot(resiso, np.ones(len(resiso)),"*", label ="transition to HL")
                #plt.plot(resisi, np.ones(len(resisi)),"*", label = "transition from HL")
                plt.legend()
                plt.xlabel(r"time(msec)")
                plt.ylabel(r"$x$")
                plt.ylim(-0.1,1.1)
                plt.title(r"g =" + GG + r" $\epsilon$ = "+ EP + " a =" + AA + r" $\sigma_x$ = " + NAMP + " tr = " + TR)
                plt.savefig('xx_g_%s_ep_%s_a_%s_ic_%s_na_%s_tr_%s.png'%(GG, EP,AA,LL,NAMP,TR))
                plt.close(fig)       
                
                
                # print(isit)
                #print(csit)
           
            # #print(fractimetemp)
            #cstimecs[k,j,l] = np.mean(uptim)
            # uptimeis[k,j,l] = np.mean(dntim)
            fractimecs[k,j,l] = np.mean(fractimetempcs)
            fractimeis[k,j,l] = np.mean(fractimetempis)
            print(k,j)
            
    
    #print(time_to_first_transition, trans_to_csstate)
    
    #print(fractimecs)
    #print(fractimeis)
    pickle.dump( time_to_first_transition, open( "timetotrancr_{0}.pkl".format(TR),"wb"))
    
    
    pickle.dump( trans_to_csstate, open( "transprobcr_{0}.pkl".format(TR),"wb"))
    
    pickle.dump( start_state_ll, open( "startstatecr_{0}.pkl".format(TR),"wb"))
    pickle.dump( fractimecs, open( "fracttimecscr_{0}.pkl".format(TR),"wb"))
    pickle.dump( fractimeis, open( "fracttimeiscr_{0}.pkl".format(TR),"wb"))
    # #pickle.dump(xdiff,  open( "xdiffghl.pkl","wb") )

import sys
if __name__ == '__main__':
    args = sys.argv
    print(args[1])
    main(args[1])


