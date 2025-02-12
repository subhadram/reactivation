#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 22:09:11 2023

@author: subhadramokashe
"""

def spike_to_rate(spikes, window_std=10):
    window_size = np.arange(-3*window_std,3*window_std,1)
    window = scipy.stats.norm.pdf(window_size, 0, window_std)
    window /= window.sum()
    n_units = spikes.shape[0]
    estimate = np.zeros_like(spikes) # Create an empty array of the same size as spikes
    for i in range(n_units):
        y = np.convolve(window, spikes[i,:], mode='same')
        estimate[i,:] = y
    return estimate