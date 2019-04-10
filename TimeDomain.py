# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:19:26 2018

@author: USER
"""


import numpy as np
import statistics


def Mean(data):
    mean = np.mean(data)
    return mean
    
def SDRR(data):
    sdrr = statistics.stdev(data)
    return sdrr

def RMSSD(data,num):
    k=0
    for i in range(0,len(data)-1):
        k += np.power(data[i]-data[i+1],2)
    k/=(num-1)
    rmssd =  np.sqrt(k)
    
    return rmssd

def RR50(data):
    datadiff = np.diff(data)
    rr50 = len(np.array([x for x in datadiff if x >= 50 or x <= (-50)]))
    prr50 = rr50/len(datadiff)*100
    return rr50, prr50

def timedomain(data):
    meanrr = np.mean(data)
    sdrr = np.std(data)
    diffrr = np.zeros(len(data)-1)
    for j in range(len(data)-1):
        diffrr[j] = data[j] - data[j+1]
    rmssd = np.sqrt(np.mean(np.power(diffrr,2)))
    sdsd = np.std(diffrr)
    nn50 = 0
    for j in range(len(data)-1):
        if np.abs(diffrr[j]) > 50:
            nn50 += 1
    pnn50 = nn50/float(len(data))*100
    return meanrr, sdrr, rmssd, sdsd, nn50, pnn50