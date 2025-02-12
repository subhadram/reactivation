#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:59:00 2024

@author: subhadramokashe
"""

from gridover_parallel import main
from multiprocessing import Pool
import numpy as np


a = np.arange(1,2,1)


if __name__ == '__main__':
    with Pool(2) as pool: # four parallel jobs
        results = pool.map(main, a)