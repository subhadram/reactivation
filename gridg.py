#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:34:37 2024

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Define constants
np.random.seed(2)

N = 500
G_L = 5e-9
C = 100e-12
E_L = -70e-3
E_K = -80e-3
V_Thresh = -50e-3

V_Reset = -65e-3
deltaT = 2e-3
tauw = 100e-3
tauw2 = 1e-3
tause = 20e-3
tausi = 5e-3

Vmax = 50e-3
dt = 1e-4
tmax = 20
tvector = np.arange(0, tmax + dt, dt)
epsilon = np.arange(0.0e-9, 0.5e-9, 0.1e-9)
trials = 1
csfrac = np.zeros((len(epsilon), trials))
isfrac = np.zeros((len(epsilon), trials))


def spikes_to_rate(spikes, dt, window):
    windowsize = int(window/dt)
    sumspike = np.zeros_like(spikes)
    i = 0
    while i < len(spikes):
        sumspike[i:i+windowsize] = np.sum(spikes[i:i+windowsize])
        i = i + windowsize
        
    sumspike = sumspike/(window)
    return sumspike

for ep in range(len(epsilon)):
    epsi = epsilon[ep]
    EP = str(epsi)
    for tr in range(trials):
        # Define network connectivity
        np.random.seed(2)  # Ensure the same random connectivity for each trial
        J = np.zeros((N, N))
        Jee1 = 1.2e-9
        Jee2 = 1.2e-9 + epsi
        Jei = 0.2e-9
        Jie = 0.8e-9
        

        overlap = 0.0
        group1 = 0.35
        group2 = 0.35
        dw0 = 1e-9
        dw2 = 2e-6
        ds = 1
        group1ei = round(group1 * N)
        group2si = round((group1 - overlap) * N)
        group2ei = round((group1 + group2 - overlap) * N)
        eind = np.zeros(N)
        eind[:group2ei] = 1
        iind = np.zeros(N)
        iind[group2ei:] = 1
        inhind = round(0.75 * N)
        J[inhind:, :inhind] = Jei
        J[:inhind, inhind:] = Jie
        J[:group1ei, :group1ei] = Jee1
        J[group2si:group2ei, group2si:group2ei] = Jee2
        pee = 0.1
        pei = 0.25
        pie = 0.5
        pii = 1
        strc = np.block([[np.random.rand(inhind, inhind) < pee,
                          np.random.rand(inhind, N - inhind) < pie],
                         [np.random.rand(N - inhind, inhind) < pei,
                          np.random.rand(N - inhind, N - inhind) < pii]])
        J = J * strc
        np.fill_diagonal(J, 0)

        dw = dw0 * eind

        # Initialize arrays
        V = np.zeros((N, len(tvector)))
        s = np.zeros((N, len(tvector)))
        w = np.zeros((N, len(tvector)))
        w2 = np.zeros((N, len(tvector)))
        spikes = np.zeros((N, len(tvector)))
        V[:, 0] = E_L + np.random.rand(N) * (V_Thresh - E_L)
        w[:, 0] = np.random.rand(N) * dw
        NE = inhind
        NI = N - NE

        # Generate random input
        gnoise_E = 0.5e-9
        gnoise_I = 0.5e-9
        rnoise_EE = 3000
        rnoise_IE = 1000
        rnoise_EI = 1800
        rnoise_II = 400
        E_E = 0
        E_I = -80e-3

        rnoise_EE_corr = 0
        rnoise_IE_corr = 0
        rnoise_EI_corr = 0
        rnoise_II_corr = 0

        tau_noiseE = 0.002
        tau_noiseI = 0.005

        f = 1
        omega = 2 * np.pi * f
        osc_rate = np.cos(omega * tvector)
        r_factor = 0.5 * np.maximum(osc_rate + 1, 0)
        plt.plot(tvector, r_factor)

        noise_EE = gnoise_E * (poisson.rvs(rnoise_EE * dt, size=(NE, len(tvector))) +
                               np.outer(np.ones(NE), poisson.rvs(rnoise_EE_corr * dt, size=len(tvector))))
        noise_EE *= (np.random.rand(NE, len(tvector)) < np.outer(np.ones(NE), r_factor))

        noise_IE = gnoise_I * (poisson.rvs(rnoise_IE * dt, size=(NE, len(tvector))) +
                               np.outer(np.ones(NE), poisson.rvs(rnoise_IE_corr * dt, size=len(tvector))))
        noise_EI = gnoise_E * (poisson.rvs(rnoise_EI * dt, size=(NI, len(tvector))) +
                               np.outer(np.ones(NI), poisson.rvs(rnoise_EI_corr * dt, size=len(tvector))))
        noise_II = gnoise_I * (poisson.rvs(rnoise_II * dt, size=(NI, len(tvector))) +
                               np.outer(np.ones(NI), poisson.rvs(rnoise_II_corr * dt, size=len(tvector))))

        gnoise_EE = np.zeros((NE, len(tvector)))
        gnoise_IE = np.zeros((NE, len(tvector)))
        gnoise_EI = np.zeros((NI, len(tvector)))
        gnoise_II = np.zeros((NI, len(tvector)))

        for i in range(1, len(tvector)):
            gnoise_EE[:, i - 1] = gnoise_EE[:, i - 1] + noise_EE[:, i - 1]
            gnoise_IE[:, i - 1] = gnoise_IE[:, i - 1] + noise_IE[:, i - 1]
            gnoise_EI[:, i - 1] = gnoise_EI[:, i - 1] + noise_EI[:, i - 1]
            gnoise_II[:, i - 1] = gnoise_II[:, i - 1] + noise_II[:, i - 1]

            gnoise_EE[:, i] = gnoise_EE[:, i - 1] * np.exp(-dt / tau_noiseE)
            gnoise_IE[:, i] = gnoise_IE[:, i - 1] * np.exp(-dt / tau_noiseI)
            gnoise_EI[:, i] = gnoise_EI[:, i - 1] * np.exp(-dt / tau_noiseE)
            gnoise_II[:, i] = gnoise_II[:, i - 1] * np.exp(-dt / tau_noiseI)

        plt.figure(4)
        plt.plot(tvector, gnoise_EE[0, :])

        # Simulate the network
        for tt in range(len(tvector) - 1):
            cellsfired = np.where(V[:, tt] >= Vmax)[0]
            cellsfirede = cellsfired[cellsfired <= group2ei]
            cellsfiredi = cellsfired[cellsfired > group2ei]

            V[cellsfired, tt] = V_Reset
            w[cellsfired, tt] = w[cellsfired, tt] + dw[cellsfired]
            w2[cellsfired, tt] = w2[cellsfired, tt] + dw2
            s[cellsfired, tt] = s[cellsfired, tt] + ds * (1 - s[cellsfired, tt])
            spikes[cellsfired, tt] = 1

            inputsfrome = np.dot(J[:, :inhind], s[:inhind, tt]) + np.concatenate((gnoise_EE[:, tt], gnoise_EI[:, tt]))
            inputsfromi = np.dot(J[:, inhind:], s[inhind:, tt]) + np.concatenate((gnoise_IE[:, tt], gnoise_II[:, tt]))

            vn = V[:, tt]
            dv1 = dt * (G_L * (E_L - vn + deltaT * np.exp((vn - V_Thresh) / deltaT)) +
                        (w[:, tt] + w2[:, tt]) * (E_K - vn) +
                        inputsfrome * (E_E - vn) + inputsfromi * (E_I - vn)) / C
            vn = V[:, tt] + 0.5 * dv1
            dv2 = dt * (G_L * (E_L - vn + deltaT * np.exp((vn - V_Thresh) / deltaT)) +
                        (w[:, tt] + w2[:, tt]) * (E_K - vn) +
                        inputsfrome * (E_E - vn) + inputsfromi * (E_I - vn)) / C
            vn = V[:, tt] + 0.5 * dv2
            dv3 = dt * (G_L * (E_L - vn + deltaT * np.exp((vn - V_Thresh) / deltaT)) +
                        (w[:, tt] + w2[:, tt]) * (E_K - vn) +
                        inputsfrome * (E_E - vn) + inputsfromi * (E_I - vn)) / C
            vn = V[:, tt] + dv3
            dv4 = dt * (G_L * (E_L - vn + deltaT * np.exp((vn - V_Thresh) / deltaT)) +
                        (w[:, tt] + w2[:, tt]) * (E_K - vn) +
                        inputsfrome * (E_E - vn) + inputsfromi * (E_I - vn)) / C

            V[:, tt + 1] = V[:, tt] + (dv1 + 2 * dv2 + 2 * dv3 + dv4) / 6
            V[:, tt + 1] = np.maximum(V[:, tt + 1], E_K)
            V[:, tt + 1] = np.minimum(V[:, tt + 1], Vmax + 1e-3)

            s[:inhind, tt + 1] = s[:inhind, tt] * np.exp(-dt / tause)
            s[inhind:, tt + 1] = s[inhind:, tt] * np.exp(-dt / tausi)
            w[:, tt + 1] = w[:, tt] * np.exp(-dt / tauw)
            w2[:, tt + 1] = w2[:, tt] * np.exp(-dt / tauw2)

        # Calculate firing rates
        spikesg1 = np.mean(spikes[:group2si, :], axis=0)
        spikesg2 = np.mean(spikes[group2si + round(overlap * N) + 1:group2ei, :], axis=0)
        spikesin = np.mean(spikes[inhind + 1:, :], axis=0)
        bin = 0.020
        rateg1 = spikes_to_rate(spikesg1, dt, bin)
        rateg2 = spikes_to_rate(spikesg2, dt, bin)
        ratein = spikes_to_rate(spikesin, dt, bin)
        fig = plt.figure(1)
        plt.plot(tvector, rateg1)
        plt.plot(tvector, rateg2)
        #plt.plot(tvector, rateov)
        plt.title(r" $\epsilon$ = "+ EP)
        plt.savefig('frate_%s.png'%(EP))
        plt.close(fig)

        rms1 = np.sqrt(np.mean(rateg1**2))
        rms2 = np.sqrt(np.mean(rateg2**2))
        rmsi = np.sqrt(np.mean(ratein**2))
        thigh1 = np.mean(rateg1 > 10)
        thigh2 = np.mean(rateg2 > 10)
        thighi = np.mean(ratein > 10)
        csfrac[ep, tr] = thigh1
        isfrac[ep, tr] = thigh2

# Plot results
plt.figure()
plt.plot(epsilon, np.mean(csfrac, axis=1))
#plt.hold(True)
plt.plot(epsilon, np.mean(isfrac, axis=1))
plt.xlabel('Epsilon')
plt.ylabel('Firing Rate')
plt.legend(['CS', 'IS'])
plt.title('Firing Rate vs Epsilon')

plt.figure()
plt.plot(tvector[9::10], 1e3 * V[0, 9::10])
#plt.hold(True)
plt.plot(tvector[9::10], 1e3 * V[N-1, 9::10])
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Membrane Potential')
plt.legend(['Neuron 1', 'Neuron ' + str(N)])
plt.show()

plt.figure()
plt.plot(tvector, rateg1, label='CS')
#plt.hold(True)
plt.plot(tvector, rateg2, label='IS')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate')
plt.title('Firing Rate')
plt.legend(['CS', 'IS'])
plt.show()
