#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:33:47 2024

@author: subhadramokashe
"""

from joblib import Parallel, delayed
def yourfunction(k1,k2):   
    s=3.14*k1*k2
    print("Area of a circle with a radius ", k1, " is:", s)

element_run = Parallel(n_jobs=-1)(delayed(yourfunction)(k,k) for k in range(1,10))