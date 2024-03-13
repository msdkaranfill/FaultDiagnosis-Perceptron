import math
import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft

# =========================================================================
# This code is mainly based on the code by System Design Optimization Lab
# (SDOL) at Korea Aerospace University (KAU)
# ============================== Input ====================================
# x : Input data
# fs : Sampling frequency
# bff : Bearing fault frequency 1x2 matrix [bpfo, bpfi]
# cutoff : Bandwidth to find amplitude of fault frequency, value was
# obtained by use of MATLAB
# ============================== Output ===================================
# feature : Calculated feature value
# feature_name: The name of features
# =========================================================================
def fault_amp(x, fs, bff, bandwidth=1017):
    # FFT
    N = len(x)
    x= np.abs(hilbert(x))
    x = x -np.mean(x)
    X = np.abs(fft(x))/N*2
    X = X[0:math.ceil(N/2)]
    f = np.arange(0, N)/N*fs
    f = f[0:math.ceil(N/2)]
    # Find amplitude at bearing fault frequency
    bpfo_ind = np.nonzero((bff[0] - bandwidth < f) & (f < bff[0] + bandwidth))
    bpfo_amp = max(X[bpfo_ind])
    bpfi_ind = np.nonzero((bff[1] - bandwidth < f) & (f < bff[1] + bandwidth))
    bpfi_amp = max(X[bpfi_ind])
    feature = np.array([bpfo_amp, bpfi_amp],dtype='float64')
    feature_name = ['BPFO', 'BPFI']
    return feature, feature_name
