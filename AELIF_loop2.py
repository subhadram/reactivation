from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import time


# Start timer
start_time = time.time()


# Cell parameters
G_L = 10e-9           # Leak conductance (S)
C = 100e-12           # Capacitance (F)
E_L = -70e-3          # Leak potential (V)
V_Thresh = -50e-3     # Threshold potential (V)
V_Reset = -80e-3      # Reset potential (V)
deltaT = 2e-3         # Threshold shift factor (V)
tau_sra = 200e-3      # Adaptation time constant (s)
a = 2e-9              # Adaptation recovery (S)
b = 0.02e-9           # Adaptation strength (A)

I0 = 0e-9             # Baseline current (A)
Vmax = 50e-3          # Voltage to detect a spike

# Simulation setup
dt = 1e-6             # Time-step in sec
tmax = 5              # Maximum time in sec
tvector = np.arange(0, tmax + dt, dt)  # Vector of all time points

ton = 0               # Time to switch on current step
toff = tmax           # Time to switch off current step
non = int(ton / dt)   # Index of time vector to switch on
noff = int(toff / dt) # Index of time vector to switch off
I = I0 * np.ones_like(tvector)  # Baseline current at all time points

Iappvec = np.arange(0.15, 0.305, 0.005) * 1e-9  # Applied current list

# Arrays to store rates and mean voltage
initialrate = np.zeros(len(Iappvec))
finalrate = np.zeros(len(Iappvec))
singlespike = np.zeros(len(Iappvec))
meanV = np.zeros(len(Iappvec))

@jit
def find_spiketimes(I):

    print(I[100])
# Initialize variables
    v = np.zeros_like(tvector)
    v[0] = E_L                     # Initial membrane potential
    I_sra = np.zeros_like(tvector)  # Adaptation current
    spikes = np.zeros_like(tvector) # Spike vector

# Time loop

    for j in range(len(tvector) - 1):
        if v[j] > Vmax:             # Spike detection
            v[j] = V_Reset          # Reset voltage
            I_sra[j] += b           # Increase adaptation current
            spikes[j] = 1           # Record the spike
    
    # Update membrane potential
        v[j+1] = v[j] + dt * ((G_L * (E_L - v[j] + deltaT * np.exp((v[j] - V_Thresh) / deltaT)) - I_sra[j] + I[j]) / C)
    
    # Update adaptation current
        I_sra[j+1] = I_sra[j] + dt * ((a * (v[j] - E_L) - I_sra[j]) / tau_sra)

# Extract spike times
    spiketimes = dt * np.where(spikes)[0]
    return spiketimes, v

# Loop over each applied current

for trial, Iapp in enumerate(Iappvec):
    print(f'{trial+1}/{len(Iappvec)}')

    # Set applied current
    I[non:noff] = Iapp
 
    [spiketimes, v] = find_spiketimes(I)
   
    # Calculate ISI rates
    if len(spiketimes) > 1:
        ISIs = np.diff(spiketimes)  # Inter-spike intervals
        initialrate[trial] = 1 / ISIs[0] if ISIs[0] != 0 else 0
        if len(ISIs) > 1:
            finalrate[trial] = 1 / ISIs[-1] if ISIs[-1] != 0 else 0
    elif len(spiketimes) == 1:
        singlespike[trial] = 1      # Record single spike if only one spike occurred
    


    meanV[trial] = np.mean(v)


# Print elapsed time
print("Elapsed time:", time.time() - start_time)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(1e9 * Iappvec, finalrate, 'k', label='Final Rate')
ISIindices = np.where(initialrate > 0)
plt.plot(1e9 * Iappvec[ISIindices], initialrate[ISIindices], 'ok', label='1/ISI(1)')
single_spike_indices = np.where(singlespike > 0)
plt.plot(1e9 * Iappvec[single_spike_indices], np.zeros(len(single_spike_indices[0])), '*k', label='Single Spike')

# Formatting plot
plt.xlabel('Iapp (nA)')
plt.ylabel('Spike Rate (Hz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

