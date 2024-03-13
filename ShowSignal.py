import os, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import hilbert
from scipy.fft import fft
from scipy.signal import lfilter, butter
from Functions.loadmatfile import load_single_example
from Functions.AR_Filter import ar_filter

# ========================= Explanation ===================================
# This code is intended to show the signal processing part in the project
# To see other files figures, simply change the name of file.
# 1. AR filter
# 2. Spectral kurtosis
# 3. Envelope analysis
# 4. display_data function
# =========================================================================
## 1. AR filter
plt.rc('font', size=10)
os.chdir('Bearing Data')
##name of the file to display
file = 'baseline.mat'
data = load_single_example(file)
fs = data['sr']                                     #sampling frequency, Hz
x = np.asarray(data['gs'][:int(fs/2)])              #part of the signal with min. length
N = len(x)
t = np.arange(0, N)/fs                              #time vector, seconds
fr = data['rate']                                   #shaft rate, Hz
bff = np.array([3.245, 4.7550])*fr                  #[BPFO BPFI]
xr, p = ar_filter(x, 700)
k1 = kurtosis(x, fisher=False)
k2 = kurtosis(xr, fisher=False)                  # Calculate kurtosis
bandwidth = 1017.2                               # optimal bandwidth
cf = 1525.9                                      # center frequency for bandpass filter
# Frequency domain signal <Figure 1>
f = np.arange(0, N)/N*fs
f = f[0:math.ceil(N/2)]
X1 = np.abs(hilbert(x))
X1 = X1 - np.mean(X1)
X1 = np.abs(fft(X1))/N*2
X1 = X1[0:math.ceil(N/2)]
X2 = np.abs(hilbert(xr))
X2 = X2 - np.mean(X2)
X2 = np.abs(fft(X2))/N*2
X2 = X2[0:math.ceil(N/2)]
plt.figure(1)
plt.subplot(211); plt.plot(f, X1); plt.xlim(0, 500); plt.ylim(0, 0.4)
plt.title(f'Envelope spectrum (Raw, Kurtosis: {k1:.4f})')
plt.axvline(bff[0], c='k', ls='--', linewidth=2)
plt.axvline(bff[1], c='r', ls='--', linewidth=2)
plt.axvline(bff[1]+fr, c='k', ls='--', linewidth=2)
plt.subplot(212); plt.plot(f, X2); plt.xlim(0, 500); plt.ylim(0, 0.4)
plt.title(f'Envelope spectrum (AR filtered, Kurtosis: {k2:.4f})')
plt.axvline(bff[1]-fr, c='k', ls='--', linewidth=2); plt.axvline(bff[1], c='r', ls='--', linewidth=2)
plt.axvline(bff[1]+fr, c='k', ls='--', linewidth=2);
plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()


def bandpass(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], 'bandpass')
    y = lfilter(b, a, data, axis=0)
    return y

xx1 = bandpass(xr, cf-bandwidth/2, cf+bandwidth/2, fs, 4)
xx2 = bandpass(xr, 100, 7000, fs, 4)

xn = np.abs(hilbert(xx1))
plt.figure(3)
plt.plot(t, xx1, c = 'gray'); plt.plot(t, xn, 'r')
plt.xlim(0, 0.5)
plt.ylim(-0.4, 0.4)
plt.xticks([]); plt.yticks([])
plt.ylabel('Envelope signal')

# envelope of signal <Figure 4>
X2 = np.abs(fft(xn-np.mean(xn)))/N; X2 = X2[0:math.ceil(N/2)]
plt.figure(4)
plt.stem(f, X2, basefmt=" "); plt.xticks([0, 200], labels=['0', '200'])
plt.xlim(0, 200)
plt.ylim(0, 0.1)
plt.axvline(bff[1]-fr, c='k', ls='--', linewidth=2, alpha=0.5)
plt.axvline(bff[1], c='r', ls='--', linewidth=2, alpha=0.5)
plt.axvline(bff[1]+fr, c='k', ls='--', linewidth=2, alpha=0.5)
plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
plt.ylim(0, 0.1)
plt.ylabel('Peak Amplitude')
plt.title(f"Envelope Spectrum: {file.split('_')[0]}")
plt.legend(["BPFO harmonics", "BPFI harmonics"], loc ="upper left")

# FFT of envelope signal <Figure 5>
plt.figure(5)
plt.stem(f, X1, basefmt=" "); plt.xticks([0, 200], labels=['0', '200'])
plt.xlim(0, 200)
plt.axvline(bff[1]-fr, c='k', ls='--', linewidth=2, alpha=0.5)
plt.axvline(bff[1], c='r', ls='--', linewidth=2, alpha=0.5)
plt.axvline(bff[1]+fr, c='k', ls='--', linewidth=2, alpha=0.5)
plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
plt.ylim(0, 0.1)
plt.ylabel('Amplitude')
plt.xlabel("Frequency (Hz)")
plt.legend(["BPFO harmonics", "BPFI harmonics"], loc="upper left")
plt.title(f"Envelope Spectrum: {file.split('_')[0]}")

# Scale-up view in(0,200)Hz of (f) <Figure 6(e)>
plt.figure(6)
plt.stem(f, X2, basefmt=" "); plt.xticks([0, 200], labels=['0', '200'])
plt.xlim(0, 200)
plt.yticks([0, 0.03, 0.06, 0.09, 0.12, 0.15])
plt.ylim(0, 0.15)
plt.ylabel('Amplitude')
plt.xlabel("Frequency (Hz)")
plt.legend(["BPFO harmonics", "BPFI harmonics"], loc="upper right")

plt.show()

