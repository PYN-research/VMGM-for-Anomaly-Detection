#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:57:03 2025

@author: user
"""
import matplotlib.pyplot as plt
x=[5,10,15,20,25,30,35,40,45,50]
y=[96.97,96.97,96.97,97.10,97.51,97.52,97.53,97.54,97.55,97.56]
y1=[94.61,94.61,94.61,94.72,95.04,95.05,95.07,95.07,95.08,95.09]
plt.plot(x,y,label='AUROC')
plt.plot(x,y1,label='AUPRC')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()