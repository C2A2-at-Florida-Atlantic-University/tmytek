import numpy as np
import matplotlib.pyplot as plt

samples_loaded = np.load('C:/Users/pwfau/OneDrive/Documents/GNURadio/Scripts/rx_samples.npy')
NUM_ANGLES = samples_loaded.shape[0]  # Number of angles in the first dimension
SIG_LENGTH = samples_loaded.shape[2]  # Length of the signal in samples
usedAngles = np.floor(np.linspace(-45, 45, NUM_ANGLES)).astype(int)

# Folder to save stuff
folder = 'C:/Users/pwfau/Downloads/in0out60/'
# File name for the saved data
cfn = 'in0out60'

rssi_file_name = folder + cfn + '_rssi.png'
snr_file_name = folder + cfn + '_snr.png'
rx_samples_file_name = folder + cfn + '_rx_samples.npy'

# Save the signal buffer to file
np.save(rx_samples_file_name, samples_loaded)

print(f"Number of angles: {NUM_ANGLES}")
print(f"Signal length: {SIG_LENGTH}")

# Initialize the rssi array
rssis = np.mean(np.abs(samples_loaded)**2, axis=2) # Average over the third dimension (time)

print(f"Mean Signal Power: {np.mean(rssis)}")
# This is the power of the noise in the signal (the power of the signal when the tx sends nothing)
noise_power = 0.00016823525947984308 
rssis_clipped = np.maximum(rssis, 2*noise_power) # Avoid negative values for SNR calculation
snr_linear = (rssis_clipped - noise_power) / noise_power # Linear SNR calculation
snrs = 10 * np.log10(snr_linear)

plt.figure(1)
extent = [-45, 45, -45, 45]  # [x_min, x_max, y_min, y_max]
plt.imshow(rssis, aspect='auto', cmap='viridis', origin='lower', 
            extent=extent, vmin=0.0, vmax=1.0)
# Set tick marks to match usedAngles
xticks = usedAngles
xlabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
plt.xticks(xticks, xlabels)
ylabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
plt.yticks(xticks, ylabels)
plt.colorbar(label='RSSI') # Add a color legend
plt.title("Beamforming Heatmap")
plt.xlabel("Rx Angle")
plt.ylabel("Tx Angle")
plt.savefig(rssi_file_name, dpi=300)
plt.tight_layout()

plt.figure(2)
plt.imshow(snrs, aspect='auto', cmap='viridis', origin='lower', 
            extent=extent)
# Set tick marks to match usedAngles
xticks = usedAngles
xlabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
plt.xticks(xticks, xlabels)
ylabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
plt.yticks(xticks, ylabels)
plt.colorbar(label='SNR (dB)') # Add a color legend
plt.title("Beamforming Heatmap")
plt.xlabel("Rx Angle")
plt.ylabel("Tx Angle")
plt.tight_layout()
plt.savefig(snr_file_name, dpi=300)
plt.show()