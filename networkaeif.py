import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib
import pickle

Ni = 10
Ne = 40
jee = 0.2 # initial e to e weight
jei = 0.1 #initial e to i connectivity
jie = -0.1 # fixed i to e connectivity
jii = -0.1 # fixed i to i connectivity
Ncells = Ni + Ne
p = 0.1

stim = np.zeros((1,4))
stim[0,0] = 1


#membrane dynamics
taue = 20 #e membrane time constant
taui = 20 #i membrane time constant
vleake = -70 #e resting potential
vleaki = -62 #i resting potential
deltathe = 2 #eif slope parameter
C = 300 #capacitance
erev = 0 #e synapse reversal potential
irev = -75 #i synapse reversal potntial
vth0 = -52 #initial spike voltage threshold
ath = 10 #increase in threshold post spike
tauth = 30 #threshold decay timescale
vre = -60 #reset potential
taurefrac = 1 #absolute refractory period
aw_adapt = 4 #adaptation parameter a
bw_adapt = .805 #adaptation parameter b
tauw_adapt = 150 #adaptation timescale

#connectivity
Ncells = Ne+Ni
tauerise = 1 #e synapse rise time
tauedecay = 6 #e synapse decay time
tauirise = .5 #i synapse rise time
tauidecay = 2 #i synapse decay time
rex = 4.5 #external input rate to e (khz)
rix = 2.25 #external input rate to i (khz)

jeemin = 1.78 #minimum ee strength
jeemax = 21.4 #maximum ee strength

jeimin = 48.7 #minimum ei strength
jeimax = 243 #maximum ei strength

jex = 1.78 #external to e strength
jix = 1.27 #external to i strength

#voltage based stdp
altd = .0008 #ltd strength
altp = .0014 #ltp strength
thetaltd = -70 #ltd voltage threshold
thetaltp = -49 #ltp voltage threshold
tauu = 10 #timescale for u variable
tauv = 7 #timescale for v variable
taux = 15 #timescale for x variable

#inhibitory stdp
tauy = 20 #width of istdp curve
eta = 1 #istdp learning rate
r0 = .003 #target rate (khz)

#populations
Npop = 2
pmem = 0.05
I = 300
Nmaxmembers = 300
#Npop = size(popmembers,1) #number of assemblies
#Nmaxmembers = size(popmembers,2) #maximum number of neurons in a population

#simulation
dt = .1 #integration timestep
T = 5 #simulatiogkn time
Nskip = 1000 #how often (in number of timesteps) to save w_in
vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset
dtnormalize = 20 #how often to normalize rows of ee weights
stdpdelay = 1 #time before stdp is activated, to allow transients to die out
Nspikes = 100 #maximum number of spikes to record per neuron



def structinitJ():
    J = np.zeros((Ncells, Ncells))
    #setting up initial weights
    J[0:Ne, 0:Ne] = jee
    J[0:Ne, Ne:Ncells] = jie
    J[Ne:Ncells, 0:Ne] = jei
    J[Ne:Ncells, Ne:Ncells] = jii
    strc = np.array(np.where(np.random.rand(Ncells, Ncells) < p,1,0))  
    J = np.multiply(J,strc)  
    return J

def make_self_connections_zero(J):
    for i in range(0,len(J)):
        J[0,0] = 0.0
    return J


# y = structinitJ()
# y = make_self_connections_zero(y)
# print(y[9,9])

#print(np.where(np.random.rand(Ncells, Ncells) < p, 1,0))



def makegroups(Npop,pmem,Nmaxmembers):
    popmembers = -1*np.ones((Npop, Nmaxmembers))
    for i in range(0,Npop):
        members = np.array(np.argwhere(np.random.rand(Ne) < pmem))
        print(members)
        popmembers[i,0:len(members)] = np.squeeze(members)
    #rint(popmembers)
    return popmembers

    

def sim(weights,popmembers):
    plt.imshow(weights)
    plt.show()
    times = np.zeros((Ncells, Nspikes)) 
    ns = np.zeros(Ncells,dtype = int)
    forwardinputse = np.zeros(Ncells) #summed weight of incoming E spikes
    forwardinputsi = np.zeros(Ncells)  #summed weight of incoming I spikes
    forwardinputseprev = np.zeros(Ncells)  #summed weight if incoming E spikes, prev time step
    forwardinputsiprev = np.zeros(Ncells) # summed weight if incoming I spikes, prev time step
    Vrec = np.zeros((Ncells,1000) )
    spikedt =  np.zeros((Ncells,1000) )
    #exponential variables
    xerise = np.zeros(Ncells)
    xedecay =  np.zeros(Ncells)
    xirise = np.zeros(Ncells)
    xidecay = np.zeros(Ncells)
    
    v = np.zeros(Ncells)
    nextx = np.zeros(Ncells) #time of next excitatory input
    sumwee0 = np.zeros(Ne) #initial summed e weight
    rx = np.zeros(Ncells) #external rate
    Nee= np.zeros(Ncells, dtype = int)#number of e->e inputs
    
    #initial values for the variables.
    for i in range(0, Ncells):
        
        v[i] = vre + (vth0-vre)*np.random.rand()
        if i <Ne:
            rx[i] = rex
            nextx[i] = -np.log(1-np.random.rand()/rx[i])
            for j in range(0,Ne):
                sumwee0[i] = sumwee0[i] + weights[i,j]
                if weights[i,j] > 0.0:
                    Nee[i] +=1
        else:
            rx[i] = rix
            nextx[i] = -np.log(1-np.random.rand()/rx[i])
    vth = vth0*np.ones(Ncells)
    wadapt = aw_adapt*(vre - vleake)*np.ones(Ncells)
    lastspike= -100*np.ones(Ncells)
    trace_istdp = np.zeros(Ncells)
    u_vstdp = np.zeros(Ne)
    v_vstdp = np.zeros(Ne)
    x_vstdp = np.zeros(Ne)
    
    Nsteps = int(T/dt)
    
    inormalize = int(dtnormalize/dt)
    print(Nsteps, inormalize)
    
    for tt in range(0, Nsteps):
        plt.imshow(weights)
        plt.show()
        Vrec[:,tt] = v
       
        t = dt*tt
        print(t)
        forwardinputse[:] = 0.0
        forwardinputsi[:] = 0.0
        
        tprev = dt*(tt-1) #only needed for the choose the simulation period.
        for ss in range(0, len(stim[:,0])):
            if (tprev<stim[ss,1]) & (t>=stim[ss,1]): #entering stimulation period
                ipop = int(stim[ss,0])
                for ii in range(0, Nmaxmembers):
                    #print(popmembers[ipop,ii])
                    if popmembers[ipop,ii] == -1:
                        #print("Hi")
                        break
                    rx[int(popmembers[ipop,ii])] += stim[ss,3] #exciting stimulation period
            if (tprev<stim[ss,2]) & (t>=stim[ss,2]):
                ipop = int(stim[ss,0])
                for ii in range(0, Nmaxmembers):
                    if int(popmembers[ipop,ii]) == -1:
                        break
                    rx[int(popmembers[ipop,ii])] -= stim[ss,3]
                    
        if tt%inormalize == 0:
            for cc in range(0,Ne):
                sumwee = 0
                for dd in range(0,Ne):
                    sumwee =+ weights[cc,dd]
                for dd in range(0,Ne):
                    if weights[cc,dd] >0.0:
                        weights[cc,dd] -= (sumwee  + sumwee0[dd]/Nee[dd])
                        if weights[cc,dd] < jeemin:
                            weights[cc,dd] = jeemin
                        elif weights[cc,dd]  >jeemax:
                            weights[cc,dd] = jeemax
        
        spiked= np.zeros(Ncells, dtype = bool)
        for cc in range(0,Ncells):
            trace_istdp[cc] -= dt*trace_istdp[cc]/tauy
            while(t > nextx[cc]):
                nextx[cc] += -np.log(1-np.random.rand()/rx[i]) #random time generation for event based simulations
                #print(t)
                if cc < Ne:
                    forwardinputseprev[cc] += jex
                else:
                    forwardinputseprev[cc] += jix
                    
            xerise[cc] += -dt*xerise[cc]/tauerise + forwardinputseprev[cc]
            xedecay[cc] += -dt*xedecay[cc]/tauedecay + forwardinputseprev[cc]
            xirise[cc] += -dt*xirise[cc]/tauirise + forwardinputsiprev[cc]
            xidecay[cc] += -dt*xidecay[cc]/tauidecay + forwardinputsiprev[cc]
            
            if cc < Ne:
                vth[cc] += dt*(vth0 - vth[cc])/tauth
                #print(vth[cc])
                wadapt[cc] += dt*(aw_adapt*(v[cc]-vleake) - wadapt[cc])/tauw_adapt
                u_vstdp[cc] += dt*(v[cc] - u_vstdp[cc])/tauu
                v_vstdp[cc] += dt*(v[cc] - v_vstdp[cc])/tauv
                x_vstdp[cc] -= dt*x_vstdp[cc]/taux
                
            if t > (lastspike[cc] + taurefrac):
                ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise)
                gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise)
                
                
                if cc < Ne:
                    dv = (vleake - v[cc] + deltathe*np.exp((v[cc]-vth[cc])/deltathe))/taue - wadapt[cc]/C + ge*(erev-v[cc])/C #+ gi*(irev-v[cc])/C 
                    v[cc] += dt*dv
                    if v[cc] > vpeak:
                        spiked[cc] = "true"
                        wadapt[cc] += bw_adapt
                else:
                    dv = (vleaki - v[cc])/taui + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C
                    v[cc] += dt*dv
                if v[cc] > vth0:
                    spiked[cc] = "true"
                    
            if spiked[cc]:
                v[cc] = vre
                lastspike[cc] = t
                ns[cc] += 1
                if ns[cc] < Nspikes:
                    times[cc,ns[cc]] = t
                trace_istdp[cc] += 1.
                if cc < Ne:
                    x_vstdp[cc] += 1. / taux
                    vth[cc] = vth0 + ath
                    
                for dd in range(0,Ncells):
                    if cc < Ne:
                        forwardinputse[dd] #+= weights[cc,dd]
                        
                    else:
                        forwardinputsi[dd] #+= weights[cc,dd]
            #print(forwardinputse)
            if spiked[cc] & (t > stdpdelay):
                if cc < Ne:
                    for dd in range( Ne,Ncells):
                        if weights[dd,cc] == 0:
                            continue
                        weights[dd,cc] += eta*trace_istdp[dd]
                        if weights[dd,cc] > jeimax:
                            weights[dd,cc] = jeimax
                else:
                    for dd in range(0,Ne):
                        if weights[cc,dd] == 0.:
                            continue
                        weights[cc,dd] += eta*(trace_istdp[dd] - 2*r0*tauy)
                        if weights[cc,dd] > jeimax:
                            weights[cc,dd] = jeimax
                        elif weights[cc,dd] < jeimin:
                            weights[cc,dd] = jeimin
            
            if spiked[cc] & (t > stdpdelay) & (cc < Ne):
                for dd in range(0,Ne):
                    continue
                if u_vstdp[dd] > thetaltd:
                    weights[cc,dd] -= altd*(u_vstdp[dd]-thetaltd)
                    if weights[cc,dd] < jeemin:
                        weights[cc,dd] = jeemin
                        
            #print(t,cc) 
            if cc < Ne:
                if (t > stdpdelay) & (v[cc] > thetaltp) & (v_vstdp[cc] > thetaltd):
                    for dd in range(0, Ne):
                        if weights[dd,cc] == 0.:
                            continue
                        weights[dd,cc] += dt*altp*x_vstdp[dd]*(v[cc] - thetaltp)*(v_vstdp[cc] - thetaltd)
                        if weights[dd,cc] > jeemax:
                            weights[dd,cc] = jeemax
        spikedt[:,tt] = spiked                
        #print(forwardinputse)               
        forwardinputseprev = forwardinputse
        forwardinputsiprev = forwardinputsi
        
        #print(forwardinputseprev, xerise)
        
        
    print(spikedt)
                            
                        
                
        
                    
                
            
        
        
                            
                        
                
                
                
            
        
    
    return Vrec,spikedt
    

J = structinitJ()
J = make_self_connections_zero(J)
groups = makegroups(Npop,pmem,Nmaxmembers)
print(groups[1])
V, spikes= sim(J,groups)
print(V.shape)
plt.plot(V[0,:])
plt.plot(V[1,:])
plt.show()

plt.imshow(1.*spikes,aspect = "auto",cmap = "gist_gray")
plt.show()