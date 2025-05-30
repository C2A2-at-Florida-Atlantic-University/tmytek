import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the 13-bit Barker code
barker = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
L = len(barker)
N = 50
TAU = N - L
rxPwr = 0.1
noisePwr = 0.01
P_FA = 0.01
P_NGP = 1e-2

snr = rxPwr / noisePwr
snr = 10 * np.log10(snr)
print(f"SNR: {snr:.2f}dB")
print(f"Power of Barker code: {np.mean(np.abs(rxPwr * barker)**2)}")
print(f"Power sqrt: {rxPwr**2}")

trials = 10000

avgPowerR = 0
avgPowerC = 0

for i in range(trials):
    noise = np.random.normal(0, np.sqrt(noisePwr), N)
    avgPowerR += np.mean(np.abs(noise)**2)
    a = np.correlate(noise, barker, mode='valid')
    avgPowerC += np.mean(np.abs(a)**2)

avgPowerR /= trials
avgPowerC /= trials

print(f"Average power of noise: {avgPowerR:.2f}")
print(f"Average power of noise correlated with Barker: {avgPowerC:.2f}")
