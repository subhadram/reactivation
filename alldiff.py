#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:47:57 2024

@author: subhadramokashe
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt

A = np.arange(0,1000,1,int)

itertools.combinations(A, 2)

k = [b- a for (a, b) in itertools.combinations(A, 2) ]


plt.hist(k)
plt.show()



#print(k)