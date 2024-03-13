from scipy.stats import kurtosis
from scipy.signal import lfilter
from nitime.algorithms.autoregressive import AR_est_YW
import numpy as np
from scipy.signal import butter, sosfilt


# ============================== Input ==========================================
# x: Vibration signal
# maxK: The maximum filter order to calculate iterations
# ix: Plot the figure when the value is entered
# ============================== Output =========================================
# ardata: Residual signal
# p: Selected filter order
# ===============================================================================

def ar_filter(x, maxk, ix = 0):
    pp = np.arange(1, maxk, 10)
    N = len(pp)
    k = np.zeros(N)
    for ix in range(0, N):
        a1 = AR_est_YW(x, pp[ix])[0]     # AR filter parameter is obtained by YW func.
        xp = lfilter(np.append(0, a1), [1], x)
        xn = x - xp
        k[ix] = kurtosis(xn, fisher=True)

    ind = np.where(k == np.max(k))[0][0]
    p = pp[ind]
    a1 = AR_est_YW(x, p)[0]
    xp = lfilter(np.append(0, a1), [1], x, axis=0)
    rx = x - xp                                         # Residual signal
    return rx, p

