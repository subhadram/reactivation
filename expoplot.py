#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:11:54 2024

@author: subhadramokashe
"""

import numpy as np
import matplotlib.pyplot as plt


timea = np.arange(0,100,0.0001)
timea2 = np.arange(-20,100,0.0001)
def expo(t,k,a0,ab):
    y = 0.8*np.exp(-k*t) + ab
    return y
    

expon = expo(timea, 0.05, 0.1, 0.2)

expon2 = np.zeros(len(timea2))
expon2[200000:]=expon
print(len(expon2), len(timea2))
#timea2 = 10+timea
#expon2 = expo(timea2, 0.1, 0.1, 0.2)
plt.plot(timea2,expon2,color = "forestgreen",linewidth = 3)
#plt.plot(timea2+20, expon2 , color = "gold",linewidth = 3)
plt.xlim(-10,40)
plt.box(False)
plt.axis(False)
plt.show()