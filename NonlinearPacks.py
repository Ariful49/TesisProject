from spectrum import aryule as _aryule
import numpy as np
from numpy import mean as _mean, dot as _dot, divide as _divide, std as _std, zeros as _zeros, append as _append, power as _power, sqrt as _sqrt, array as _array, log
from math import cos as _cos, sin as _sin, atan as _atan, degrees as _degrees, radians as _radians


def _atand(x):
    return _degrees(_atan(x))
def _cosd(x):
    return _cos(_radians(x))
def _sind(x):
    return _sin(_radians(x))

def _embed_seq(data,tau,de):
    N = len(data)
    if de*tau>N:
        raise ValueError ("D*Tau > N")
    if tau<1:
        raise ValueError ("Tau has to be at least 1")
    Y = np.zeros((de, N - (de-1)*tau))
    
    for i in range(de):
        Y[i] = data[i*tau : i*tau + Y.shape[1]]
    return Y.T

def PUCK(data,size_data,k,M):
    N = size_data
    if len(data) < N+k+M:
        l = len(data)+k+M-len(data)
        data = _append(data,data[0:l])
    AR, p, k2 = _aryule(data,k)
    
    I = _zeros(N+M+1)
    for t in range(0,N+M+1):
        for n in range(0,k):
            I[t] = I[t]+AR[n]*data[t+n]
    
    Im = _zeros(N)
    for t in range(N):
        for n in range(M):
            Im[t] = Im[t]+I[t+n]
        Im[t] = Im[t]/float(M)
    
    I = I[M:N+M+1]
    B = [I[t]-Im[t] for t in range(N)]
    C = [I[t+1]-I[t] for t in range(N)]
    centroid = [_mean(B), _mean(C)]
    avgd = _mean(_sqrt(_power(abs(B-centroid[0]),2)+_power(abs(C-centroid[1]),2)))
    #distance = np.sqrt(np.power(np.abs(B-centroid[0,0]),2)+np.power(np.abs(C-centroid[0,1]),2))
    w = 1
    slope = _divide(_dot(_dot(B,w),C),_dot(_dot(B,w),B))
    
    x1 = B[0]
    y1 = _dot(slope,x1)
    x2 = B[N-1]
    y2 = _dot(slope,x2)
    slopeAngle = _atand((y1-y2)/(x2-x1))
    rotate = _dot([[_cosd(slopeAngle), -_sind(slopeAngle)], [_sind(slopeAngle), _cosd(slopeAngle)]], [B[M:N-2], C[M:N-2]])
    new_interval = rotate[0]
    new_time = rotate[1]
    ssd1 = _std(new_time)
    ssd2 = _std(new_interval)
    
    #slope apakah menggambar x1 dan y1 nya?
    #plot poincare
  
    return slope, ssd1, ssd2, avgd, B ,C

def coarse_grainning (data, tau):
    if tau ==1:
        return data
    lenght_out = int(data.size / tau)
    
    #n_dropped = data.size % tau
    #mat = data[0:data.size - int(n_dropped)].reshape((tau, int(lenght_out)))
    #return np.mean(mat, axis=0)
    
    data_list = []
    
    for j in range(1,lenght_out):
        hitung = 0
        for i in range(int((j-1)*tau)+1, int(j*tau)):
            hitung += data[i]
        hitung /= tau
        data_list.append(hitung)
    
    return data_list    
        
    
def SampEn(time_series, m, r):

    def _maxdist(x_i, x_j):
        result = max([abs(ua - va) for ua, va in zip(x_i, x_j)]) #define jarak
        return result

    def _phi(m):
        x = [[time_series[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)] #bikin vektor xi dan xj
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))] #perhitungan phi
        return sum(C)

    N = len(time_series)
    
    return -log(_phi(m+1) / float(_phi(m))) #harus di float khusus phyton 2 dan -log dibalik dari ln

def ApEn(time_series, m, r):

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)]) #define jarak

    def _phi(m): 
        x = [[time_series[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)] #ini bikin vektorny
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(log(C)) #ini adalah rumus ApEn tapi cuman satu sisi
    N = len(time_series)

    return abs(_phi(m+1) - _phi(m)) #ini rumus ApEN yang m+1 - m

def Poincare(x):
    rotAngle = -45
    xaxis, yaxis = x[0:len(x)-1], x[1:len(x)]
    rot_matrix = _dot([[_cosd(rotAngle), -_sind(rotAngle)],[_sind(rotAngle), _cosd(rotAngle)]],[xaxis,yaxis])
    pSD1 = _std(rot_matrix[1])
    pSD2 = _std(rot_matrix[0])
    return pSD1, pSD2

def DFA(data):
    def _DFA(data,win_length,order):
        N = len(data)
        n = int(np.floor(N/float(win_length)))
        N1 = int(n*win_length)
        Yn = np.zeros(N1)
        mn = np.mean(data[0:N1])
        y = [np.sum(data[0:i+1]-mn) for i in range(N1)]
        for j in range(n):
            fitcoef = np.polyfit(np.arange(win_length),y[j*win_length:(j+1)*win_length],order)
            Yn[j*win_length:(j+1)*win_length] = np.polyval(fitcoef,np.arange(win_length))
        return np.sqrt(np.sum(np.power((y-Yn),2))/float(N1))

    def _DFA2(data,start,stop):
        n = np.arange(start,stop,1)
        N1 = len(n)
        F_n = np.zeros(N1)
        for i in range(N1):
            F_n[i] = _DFA(data,int(n[i]),1)
        A = np.polyfit(np.log10(n),np.log10(F_n),1)
        alpha = A[0]
        D = 3-A[0]
        return n, F_n, D, alpha
    
    _,_,_,alpha1 = _DFA2(data,4,12)
    _,_,_,alpha2 = _DFA2(data,11,len(data))
    return alpha1, alpha2

def ToneEntropy(data):
    diffrr = np.zeros(len(data)-1)
    for j in range(len(data)-1):
        diffrr[j] = data[j] - data[j+1]
    for k in range(len(data)):
        p = np.mean((diffrr/data[k])*100)
    S = -(np.sum(p * np.log(p), axis=0))
    return S

def samp_entropy(data, m, r, tau=1, relative_r = True):
    coarse_a = coarse_grainning(data, tau)
    if relative_r:
        coarse_a /= np.std(coarse_a)
    embsp = _embed_seq (coarse_a, 1, m+1)
    embsp_last = embsp[:,-1]
    embs_mini = embsp[:,:-1]
    # Buffers are preallocated chunks of memory storing temporary results.
    # see the `out` argument in numpy *ufun* documentation    
    dist_buffer = np.zeros(np.shape(embsp)[0]-1, dtype = np.float32)
    subtract_buffer = np.zeros((dist_buffer.size,m), dtype = np.float32)
    in_range_buffer = np.zeros_like(dist_buffer, dtype = np.bool)
    sum_cm, sum_cmp = 0.0, 0.0
    
    # we iterate through all templates (rows), except last one.
    for i,template in enumerate(embs_mini[:-1]):

        # these are just views to the buffer arrays. to store intermediary matrices
        dist_b_view = dist_buffer[i:]
        sub_b_view = subtract_buffer[i:]
        range_b_view = in_range_buffer[i:]
        embsp_view = embsp_last[i+1:]

        # substract the template from each subsequent row of the embedded matrix
        np.subtract(embs_mini[i+1:],  template, out=sub_b_view)
        # Absolute distance
        np.abs(sub_b_view, out=sub_b_view)
        # Maximal absolute difference between a scroll and a template is the distance
        np.max(sub_b_view, axis=1, out=dist_b_view)
        # we compare this distance to a tolerance r
        np.less_equal(dist_b_view, r, out= range_b_view)
        # score one for this template for each match
        in_range_sum = np.sum(range_b_view)
        sum_cm  += in_range_sum

        ### reuse the buffers for last column
        dist_b_view = dist_buffer[:in_range_sum]

        where = np.flatnonzero(range_b_view)
        dist_b_view= np.take(embsp_view,where,out=dist_b_view)
        range_b_view = in_range_buffer[range_b_view]
        # score one to TODO for each match of the last element
        dist_b_view -= embsp_last[i]
        np.abs(dist_b_view, out=dist_b_view)
        np.less_equal(dist_b_view, r, out=range_b_view)
        sum_cmp += np.sum(range_b_view)

    if sum_cm == 0 or sum_cmp ==0:
        return np.NaN
    return np.log(sum_cm/sum_cmp)
    

'''
data = [DATAMU]
for i in range(len(data)):
    meanrr, sdrr, rmssd, sdsd, nn50, pnn50 = timedomain(data[i,:])
    vlf,lf,hf,tp,lfhf,lfnorm,hfnorm = freqdomain(data[i,:])
    alpha1, alpha2 = nonlin(data[i,:])
'''