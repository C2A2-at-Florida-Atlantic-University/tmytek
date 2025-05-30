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
P_FA = 0.001
P_NGP = 1e-2

snr = rxPwr / noisePwr
snr = 10 * np.log10(snr)
print(f"SNR: {snr:.2f}dB")
print(f"Power of Barker code: {np.mean(np.abs(rxPwr * barker)**2)}")
print(f"Power sqrt: {rxPwr**2}")

alphaSquared = rxPwr

M = (L - 1) 

t0 = rxPwr + norm.ppf(1 - P_FA) * np.sqrt(noisePwr * L)

prob = norm.cdf(((L-1)*alphaSquared)/(np.sqrt(2*L) * np.sqrt(noisePwr))) * 100
snrFloor = (np.sqrt(2*L) * norm.ppf(1 - P_NGP)) / (L-1)
snrFloor = 10 * np.log10(snrFloor)
print(f"Threshold t0: {t0}")
print(f"Probability thresh should be bigger: {prob:.0f}%")
print(f"SNR floor such that max peak is greater than other peaks {100*(1-P_NGP):.4f}% of the time: {snrFloor:.2f}dB")

fNegatives = 0
fPositives = 0
tNegatives = 0
tPositives = 0

avgDistToPeak = 0
closestDistToPeak = 0

trials = 10000

for i in range(trials):
    sig = np.zeros(N)
    TAU = np.random.randint(0, N+L*2)
    for j in range(L):
        if j + TAU < N:
            sig[j + TAU] = barker[j] * np.sqrt(rxPwr)
        else:
            break
    noise = np.random.normal(loc=0, scale=np.sqrt(noisePwr), size=N)
    sig += noise
    marked_positive = False
    a = np.correlate(sig, barker, mode='valid')
    if False:
      plt.figure(1)
      plt.plot(sig, 'r')
      plt.figure(2)
      plt.plot(a, 'b')
      plt.axhline(y=t0, color='r', linestyle='--', label='Threshold')
      plt.axvline(x=TAU, color='b', linestyle='--', label='Tau')
      plt.show()
    max_index = np.argmax(a)
    max_peak = a[max_index]
    if max_peak > t0:
      marked_positive = True
      if max_index == TAU:
          avgDistToPeak += max_peak - t0
          if max_peak - t0 < closestDistToPeak or closestDistToPeak == 0:
              closestDistToPeak = max_peak - t0
          tPositives += 1
      else:
          fPositives += 1
          if False:
            plt.figure(1)
            plt.plot(sig, 'r')
            plt.figure(2)
            plt.plot(a, 'b')
            plt.axhline(y=t0, color='r', linestyle='--', label='Threshold')
            plt.axvline(x=TAU, color='b', linestyle='--', label='Tau')
            plt.show()
            exit(0)
          
    corrs_greater_than_t0 = np.where(a > t0)[0]
    if not marked_positive:
        if TAU > N - L:
            tNegatives += 1
        else:
            fNegatives += 1


print(trials)
print(f"False Positives: {fPositives} / {trials} = {100 * fPositives / trials:.2f}%")
print(f"False Negatives: {fNegatives} / {trials} = {100 * fNegatives / trials:.2f}%")
print(f"True Positives: {tPositives} / {trials} = {100 * tPositives / trials:.2f}%")
print(f"True Negatives: {tNegatives} / {trials} = {100 * tNegatives / trials:.2f}%")

acc = (tPositives + tNegatives) / trials
print(f"Accuracy: {acc:.2f}")
recall = tPositives / (tPositives + fNegatives)
print(f"Recall: {recall:.2f}")
if tPositives > 0:
  avgDistToPeak /= tPositives
  print(f"Avg distance to peak: {avgDistToPeak:.2f}")
print(f"Closest distance to peak: {closestDistToPeak:.2f}")


cnt = 0

for i in range(trials):
    v = np.random.normal(loc=0, scale=np.sqrt(noisePwr)) + np.sqrt(alphaSquared)
    if v > t0:
        cnt += 1

print(f"Probability of false alarm: {cnt / trials:.4f}")
exit(0)


if corrs_greater_than_t0.size > 0:
        first_peak_index = corrs_greater_than_t0[0]
        last_acceptable_peak_index = first_peak_index + (L * 3) // 2
        if N -last_acceptable_peak_index > L // 3:
            max_peak_index = np.argmax(a[first_peak_index:last_acceptable_peak_index]) + first_peak_index
            max_peak = a[max_peak_index]
            if max_peak > t0:
                marked_positive = True
                if max_peak_index == TAU:
                    tPositives += 1
                else:
                    fPositives += 1
sig = np.zeros(N)
for i in range(L):
    if i + TAU < N:
        sig[i + TAU] = barker[i] * rxPwr
    else:
        break
plt.figure(1)
plt.plot(sig, 'r')
a = np.correlate(sig, barker, mode='valid')
print(f"Max Correlation: {np.max(a)}")
#print(a[TAU])
plt.figure(2)
plt.plot(a, 'b')
corr_vals = np.zeros(N+L)
for tau in range (N+L):
    sig = np.zeros(N)
    for i in range(L):
        if i + tau < N:
            sig[i + tau] = barker[i] * rxPwr
        else:
            break
    a = np.correlate(sig, barker, mode='valid')
    max_corr = np.max(a)
    corr_vals[tau] = max_corr
plt.figure(3)
plt.plot(corr_vals, 'g')
plt.show()