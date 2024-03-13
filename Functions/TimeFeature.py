import numpy as np
from numpy import std, abs, sum, sqrt, max, min, array
from scipy.stats import kurtosis, skew

def timefeature(x):
     """returns time feature values and names of the given signal data"""

     rms0 = 0.86256                               #maen rms of baseline signals for normalization
     N = len(x)
     xm = sum(x) / N                                                            # 1. Mean
     xsd = std(x, ddof=1, axis=0)                                               # 2. Standard deviation
     xrms = sqrt(sum(x ** 2)/len(x))                                            # 3. RMS
     xsk = skew(x, axis=0)                                                      # 4. Skewness
     xkurt = kurtosis(x, fisher=False)                                          # 5. Kurtosis
     xsf = xrms / (1 / N * sum(abs(x)))                                         # 6. Shape factor
     xcf = max(abs(x)) / xrms                                                   # 7. Crest factor
     xif = max(abs(x)) / (1 / N * sum(abs(x)))                                  # 8. Impulse factor
     xmf = max(abs(x)) / ((sum(sqrt(abs(x))) / N) ** 2)                         # 9. Margin factor
     xp = max(abs(x))                                                           # 10. Peak
     xp2p = max(x) - min(x)                                                     # 11. Peak-to-peak
     talaf = np.log(xkurt+(xrms/rms0))                                          # 12. TALAF
     thikat = np.log(xkurt**xcf+((xrms/rms0)**xp))                              # 13. THIKAT

     feature = array([xm, xsd, xrms, xsk, xkurt, xsf, xcf, xif, xmf, xp, xp2p, talaf, thikat],dtype='float64')
     feature_name = ['MEAN','STD','RMS','SK','KUR','SF','CF','IF','MF','PEAK','P2P', 'TALAF', 'THIKAT']
     return feature, feature_name

