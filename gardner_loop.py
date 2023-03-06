import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


df = pd.read_csv('new.csv', header=None, names=['count', 'power'])



# Extract the signal column from the dataframe
signal = df['power'].values

# Normalize the signal
signal = (signal - np.mean(signal)) / np.std(signal)

# Apply a bandpass filter to extract the signal at the desired frequency range
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

fs = 1000  # Sample rate
lowcut = 40  # Lower cutoff frequency
highcut = 60  # Upper cutoff frequency
b, a = butter_bandpass(lowcut, highcut, fs, order=5)
signal_filt = filtfilt(b, a, signal)

# Implement the Gardner loop algorithm to estimate the symbol rate of the signal
n = len(signal_filt)
T = 2*np.pi/highcut  # Period of the signal
tau = int(np.round(T*fs))  # Number of samples per period
mu = 0.05  # Loop gain
theta_hat = np.zeros(n)
phi_hat = np.zeros(n)
theta = np.zeros(n)
phi = np.zeros(n)
error = np.zeros(n)

for i in range(tau, n):
    theta[i] = theta[i-1] + 2*np.pi/T - mu*error[i-tau]*(phi[i-tau]-phi_hat[i-tau])
    phi[i] = phi[i-1] + theta[i] - theta_hat[i-1]
    theta_hat[i] = theta_hat[i-1] + 2*np.pi/T - mu*error[i-tau]*(phi_hat[i-tau]-phi_hat[i-tau-1])
    phi_hat[i] = phi_hat[i-1] + theta_hat[i] - theta_hat[i-1]
    error[i] = signal_filt[i-tau] * np.sin(phi[i]-phi_hat[i])

# Plot the results
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(df['count'], signal)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Signal')
ax[0].set_title('Original Signal')
ax[1].plot(df['count'], signal_filt)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Signal')
ax[1].set_title('Filtered Signal')
ax[2].plot(df['count'][tau:], error[tau:])
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Error')
ax[2].set_title('Loop Error')
plt.tight_layout()
plt.show()