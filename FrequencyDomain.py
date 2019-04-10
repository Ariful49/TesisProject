# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:18:47 2018

@author: USER
"""
from scipy.fftpack import fft
import numpy as np
'''
def frequency(data):
    yf = abs(fft(data))  #atau dibagi Npoints 
    vlf = np.sum(yf[0:int(np.ceil(0.04*len(data)))])
    lf = np.sum(yf[int(np.ceil(0.04*len(data)))+1:int(np.ceil(0.15*len(data)))])
    hf = np.sum(yf[int(np.ceil(0.15*len(data)))+1:int(np.ceil(0.4*len(data)))])
    tp = np.sum(yf[0:int(np.ceil(0.4*len(data)))])
    lfhf = lf/hf
    
    return vlf, lf, hf, tp, lfhf
'''

def freqdomain(data):
    def nextpow2(i):
        n = 1
        while n < i: n *= 2
        return n
    
    nfft = nextpow2(len(data))
    psd = 1/(nfft/2.) * fft(data,nfft)*np.conj(fft(data,nfft))
    vlf = np.sum(psd[2:40])
    lf = np.sum(psd[40:153])
    hf = np.sum(psd[153:409])
    tp = np.sum(psd[2:409])
    lfhf = lf/hf
    lfnorm = lf/(lf+hf)*100
    hfnorm = hf/(lf+hf)*100
    return nfft, vlf,lf,hf,tp,lfhf,lfnorm,hfnorm


def freqdomainnew(data):
    def nextpow2new(i):
        n = 1
        while n < i: n *= 2
        return n
    #ini range nya dibagi 10, karena 10 Hz
    nfftnew = nextpow2new(len(data))
    psdnew = 1/(nfftnew/2.) * fft(data,nfftnew)*np.conj(fft(data,nfftnew)) #128*16, 256*16, 256*16
    #psdnew = 1/(nfftnew/2.) * np.power(fft(data,nfftnew),2)
    vlfnew = np.sum(psdnew[2:40]) #1-16
    lfnew = np.sum(psdnew[40:153]) #17-61
    hfnew = np.sum(psdnew[153:409]) #62-164
    tpnew = np.sum(psdnew[2:409]) #1-164
    #vlfnew = np.sum(psdnew[13:163])
    #lfnew = np.sum(psdnew[163:613])
    #hfnew = np.sum(psdnew[613:1637])
    #tpnew = np.sum(psdnew[13:1637])
    lfhfnew = lfnew/hfnew
    lfnormnew = lfnew/(lfnew+hfnew)*100
    hfnormnew = hfnew/(lfnew+hfnew)*100
    return nfftnew, vlfnew,lfnew,hfnew,tpnew,lfhfnew,lfnormnew,hfnormnew



'''
data = [DATAMU]
for i in range(len(data)):
    meanrr, sdrr, rmssd, sdsd, nn50, pnn50 = timedomain(data[i,:])
    vlf,lf,hf,tp,lfhf,lfnorm,hfnorm = freqdomain(data[i,:])
    alpha1, alpha2 = nonlin(data[i,:])
'''