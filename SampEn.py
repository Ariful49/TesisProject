import csv
import pandas as pd
import numpy as np

ecg = pd.read_csv("E:/ecg-test.csv", sep=';')

def SampEn(U, m, r):
    def _maxdist(x_i, x_j):
        result = max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        return result
    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = 1.*np.array([len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))])
        return sum(C)
    N = len(U)
    return -np.log(_phi(m+1) / _phi(m))

SampEn(ecg['heartrate'],2,3)