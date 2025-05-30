import numpy as np
import matplotlib.pyplot as plt
import TxRxBase

TxRxBase.SAMPLE_RATE = 1000000 # Sample rate in Hz
TxRxBase.setUsedAngles(30 + 1) # Number of angles in the index book
TxRxBase.SIG_LENGTH = 256 * 2 - 13 # Length of the signal in samples
TxRxBase.DELAY_BETWEEN_PACKETS_MS = 10 # MAX delay between packets in milliseconds
TxRxBase.TMYTEK_BF_SETTLE_TIME_MS = 1 # Settle time for TMYTEK beamforming in milliseconds

# BF angle index for both tx and rx
angle_index = 0
# Signal powers for each angle
signal_powers = np.zeros((TxRxBase.NUM_ANGLES, TxRxBase.NUM_ANGLES), dtype=np.float32)
# Initialize the 3D array (Rx x Tx x Signal Length)
rx_samples = np.zeros((TxRxBase.NUM_ANGLES, TxRxBase.NUM_ANGLES, TxRxBase.SIG_LENGTH), dtype=np.complex64)

# Function that will be called when a packet is received properly
def rx_callback(sig):
    global angle_index, signal_powers, rx_samples
    tx_angle_index = int(np.floor(angle_index / TxRxBase.NUM_ANGLES))
    rx_angle_index = angle_index % TxRxBase.NUM_ANGLES
    signal_powers[tx_angle_index, rx_angle_index] = np.mean(np.abs(sig)**2)
    rx_samples[tx_angle_index, rx_angle_index, :] = sig[13:] # Remove the Barker code

# Start the UDP receiver for the GNU Radio sink
TxRxBase.start_rx_udp(rx_callback)

# Generate time vector
t = np.arange(TxRxBase.SIG_LENGTH) / TxRxBase.SAMPLE_RATE
# Generate Zadoff-Chu sequence
n = np.arange(TxRxBase.SIG_LENGTH)
zc_sequence = np.exp(-1j * np.pi * n * (n + 1) / (TxRxBase.SIG_LENGTH))

while True:
    tx_angle_index = int(np.floor(angle_index / TxRxBase.NUM_ANGLES))
    rx_angle_index = angle_index % TxRxBase.NUM_ANGLES
    TxRxBase.transmit(zc_sequence, TxRxBase.usedAngles[tx_angle_index], TxRxBase.usedAngles[rx_angle_index])
    angle_index = angle_index + 1
    if angle_index >= TxRxBase.NUM_ANGLES * TxRxBase.NUM_ANGLES: # If reached max angle index
        angle_index = 0 # Reset angle index
        np.save('C:/Users/pwfau/OneDrive/Documents/GNURadio/Scripts/rx_samples.npy', rx_samples)
        plt.figure()
        extent = [-45, 45, -45, 45]  # [x_min, x_max, y_min, y_max]
        plt.imshow(signal_powers, aspect='auto', cmap='viridis', origin='lower', 
                    extent=extent, vmin=0.0, vmax=1.0)
        # Set tick marks to match usedAngles
        xticks = TxRxBase.usedAngles
        xlabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
        plt.xticks(xticks, xlabels)
        ylabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
        plt.yticks(xticks, ylabels)
        plt.colorbar(label='RSSI') # Add a color legend
        plt.title("Beamforming Heatmap")
        plt.xlabel("Rx Angle")
        plt.ylabel("Tx Angle")
        plt.tight_layout()
        plt.show()
        signal_powers = np.zeros((TxRxBase.NUM_ANGLES, TxRxBase.NUM_ANGLES), dtype=np.float32)