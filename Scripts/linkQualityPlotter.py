import numpy as np
import os
import matplotlib.pyplot as plt

dict_loaded = np.load('C:/Users/pwfau/OneDrive/Documents/GNURadio/Scripts/rx_samples.npy', allow_pickle=True).item()
rx_samples = dict_loaded['rx_samples']
rx_samples_downsampled = dict_loaded['rx_samples_downsampled']
seed = dict_loaded['seed']
rng = np.random.default_rng(seed) # Random number generator for PSK modulation
NUM_ANGLES = rx_samples.shape[0]  # Number of angles in the first dimension
SIG_LENGTH = rx_samples.shape[2]  # Length of the signal in samples
usedAngles = np.floor(np.linspace(-45, 45, NUM_ANGLES)).astype(int)
NUM_SCHEMES = 3
SCHEMES = [2, 4, 8] # Number of schemes for PSK modulation
schemeNames = ['ZC', 'BPSK', 'QPSK', '8PSK']
SPS = 4 # Samples per symbol
k = int((SIG_LENGTH-13) / SPS)

# Folder to save stuff
folder = 'C:/Users/pwfau/Downloads/in0out60/'
# Folder name for the XRIFLE used
xrfile = 'in0out60/'

directory = folder + xrfile

if not os.path.exists(directory):
    os.makedirs(directory)  # Create the directory if it doesn't exist

# Save the signal buffer to file
np.save(directory + "rx_samples.npy", dict_loaded)

# This is the power of the noise in the signal (the power of the signal when the tx sends nothing)
noise_power = 0.00016823525947984308

rssis = np.mean(np.abs(rx_samples)**2, axis=3, keepdims=True)  # Shape: (Rx, Tx, Scheme, 1)
rssis_clipped = np.maximum(rssis, 2*noise_power) # Avoid negative values for SNR calculation
snr_linear = (rssis_clipped - noise_power) / noise_power # Linear SNR calculation
snrs = 10 * np.log10(snr_linear)

for scheme_index in range(NUM_SCHEMES):
    M = SCHEMES[scheme_index]
    syms = rng.integers(0, M, size=k) # Generate random symbols
    phase_offset = np.pi / M if M > 2 else 0  # π/4 for QPSK, π/8 for 8-PSK, 0 for BPSK
    tx_sig = np.exp(1j * (2 * np.pi * syms / M + phase_offset))
    rx_sig = rx_samples_downsampled[:, :, scheme_index, :]

for scheme_index in range(NUM_SCHEMES + 1):
    plt.figure(scheme_index * 2 + 1)
    extent = [-45, 45, -45, 45]  # [x_min, x_max, y_min, y_max]
    plt.imshow(rssis[:, :, scheme_index], aspect='auto', cmap='viridis', origin='lower', 
                extent=extent, vmin=0.0, vmax=1.0)
    # Set tick marks to match usedAngles
    xticks = usedAngles
    xlabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
    plt.xticks(xticks, xlabels)
    ylabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
    plt.yticks(xticks, ylabels)
    plt.colorbar(label='RSSI') # Add a color legend
    plt.title(f"Beamforming Heatmap {schemeNames[scheme_index]}")
    plt.xlabel("Rx Angle Index")
    plt.ylabel("Tx Angle Index")
    plt.tight_layout()
    plt.savefig(directory+schemeNames[scheme_index]+"rssi.png", dpi=300)

    plt.figure(scheme_index * 2 + 2)
    plt.imshow(snrs[:, :, scheme_index], aspect='auto', cmap='viridis', origin='lower', 
                extent=extent)
    # Set tick marks to match usedAngles
    xticks = usedAngles
    xlabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
    plt.xticks(xticks, xlabels)
    ylabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
    plt.yticks(xticks, ylabels)
    plt.colorbar(label='SNR (dB)') # Add a color legend
    plt.title(f"Beamforming Heatmap {schemeNames[scheme_index]}")
    plt.xlabel("Rx Angle Index")
    plt.ylabel("Tx Angle Index")
    plt.tight_layout()
    plt.savefig(directory+schemeNames[scheme_index]+"snr.png", dpi=300)


plt.show()