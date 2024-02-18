import numpy as np

def convolution(a:np.ndarray, b:np.ndarray):
    # switch to let 'b' is shorter
    if a.size < b.size:
        temp = a 
        a = b
        b = temp
    n_a = a.size
    n_b = b.size
    # want to multiple M[n_a+n_b-1, n_b] * b [n_b, 1] = y[n_b+n_a-1, 1]
    # make the matrix M[n_a+n_b-1, n_b]
    M = np.zeros((n_a + n_b -1 ,n_b))
    for i in range(n_a+n_b-1):
        # value of the first colume 
        if i < n_a:
            M[i,0] = a[i]
        else:
            M[i,0] = 0
        
        # the value of the i-th row
        for j in range(1,n_b):
            # the first row
            if i == 0:
                M[i,j] = 0
            # the other row is equal to previous row but add one element at top
            else:
                M[i,j] = M[i-1,j-1]

    y = np.matmul(M, np.transpose(b))
    return np.transpose(y)

def findPeaks(signal, thersold, wide)->list:
    # store the point whose value is expectially high
    # campare with the value store in the buffer 
    peak = []
    buffer = [0]*wide + [0] + list(signal[0 : wide])

    ### main ###
    for i in range(len(signal)):
        if (i + wide) < len(signal):
            buffer.remove(buffer[0])
            buffer.append(signal[i + wide])
        y = signal[i]
        ## determine the point is the alert point
        if y > thersold and y >= max(buffer):
            peak.append(i) # the n of the onsets
    return peak

def find2Peaks(signal, thresholdHigh, thresholdLow, wide, **kwargs):
    # store the point whose value is expectially high
    # campare with the value store in the buffer 
    alert = []
    alert2 = []
    buffer = [0]*wide + [0] + list(signal[0 : wide])

    ### main ###
    for i in range(len(signal)):
        if (i + wide) < len(signal):
            buffer.remove(buffer[0])
            buffer.append(signal[i + wide])
        y = signal[i]
        ## determine the point is the alert point
        if y > thresholdHigh and y >= max(buffer):
            alert.append(i) # the n of the onsets
        elif y > thresholdLow and y >= max(buffer):
            alert2.append(i)
    return (alert,alert2)

def findFirstPeak(signal)->list:
    # store the point whose value is expectially high
    # campare with the value store in the buffer
    wide = max(len(signal) // 20, 10)
    # buffer = [0]*wide + [0] + list(signal[0 : wide])
    buffer = list(signal[0: 2*wide])
    ### main ###
    for i in range(wide, len(signal)):
        if (i + wide) < len(signal):
            buffer.remove(buffer[0])
            buffer.append(signal[i + wide])
        y = signal[i]
        ## determine the point is the alert point
        if y > np.std(buffer) and y >= max(buffer):
            return i # the x[n] of the signal
    return 0
