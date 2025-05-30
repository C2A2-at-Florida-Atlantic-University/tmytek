import numpy as np
import matplotlib.pyplot as plt
import socket
import random
import TxRxBase

GNU_UDP_RX_CONST_GUI_PORT = 5006 # Port for the Rx GUI to receive constalltion data
GNU_UDP_TX_CONST_GUI_PORT = 5008 # Port for the Tx GUI to receive constalltion data

TxRxBase.SAMPLE_RATE = 1000000 # Sample rate in Hz
TxRxBase.setUsedAngles(30 + 1) # Number of angles in the index book
TxRxBase.SIG_LENGTH = 256 * 2 - 13 # Length of the signal in samples
TxRxBase.DELAY_BETWEEN_PACKETS_MS = 10 # MAX delay between packets in milliseconds
TxRxBase.TMYTEK_BF_SETTLE_TIME_MS = 1 # Settle time for TMYTEK beamforming in milliseconds

filt = TxRxBase.rrcFilter # Filter for pulse shaping

sock3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # GUI rx constellation sock 
sock4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # GUI rx constellation sock 

NUM_SCHEMES = 3
SCHEMES = [2, 4, 8] # Number of schemes for PSK modulation
scheme_index = 0 # Index for the current scheme

# BF angle index for both tx and rx
angle_index = 0
# Initialize the 3D array (Rx x Tx x Scheme x Signal Length)
rx_samples = np.zeros((TxRxBase.NUM_ANGLES, TxRxBase.NUM_ANGLES, NUM_SCHEMES+1, TxRxBase.SIG_LENGTH), dtype=np.complex64)
rx_samples_downsampled = np.zeros((TxRxBase.NUM_ANGLES, TxRxBase.NUM_ANGLES, NUM_SCHEMES, TxRxBase.SIG_LENGTH//TxRxBase.SPS), dtype=np.complex64)

seed = np.random.SeedSequence().entropy # Generate a random seed
rng = np.random.default_rng(seed) # Random number generator for PSK modulation

# Function that will be called when a packet is received properly
def rx_callback(sig):
    global angle_index, rx_samples, scheme_index
    tx_angle_index = int(np.floor(angle_index / TxRxBase.NUM_ANGLES))
    rx_angle_index = angle_index % TxRxBase.NUM_ANGLES
    rx_samples[tx_angle_index, rx_angle_index, scheme_index, :] = sig[13:]
    if scheme_index > 0:
        sig = sig[13:] # Remove the Barker code
        sig = TxRxBase.pulseShapeRx(sig, filt)
        rx_samples_downsampled[tx_angle_index, rx_angle_index, scheme_index-1, :] = sig
        dataToSend = np.array(sig)
        dataToSend = TxRxBase.encode_data(dataToSend) # Convert to bytes
        sock3.sendto(dataToSend, ('localhost', GNU_UDP_RX_CONST_GUI_PORT))

# Start the UDP receiver for the GNU Radio sink
TxRxBase.start_rx_udp(rx_callback)

# Generate time vector
t = np.arange(TxRxBase.SIG_LENGTH) / TxRxBase.SAMPLE_RATE
# Generate Zadoff-Chu sequence
n = np.arange(TxRxBase.SIG_LENGTH)
zc_sequence = np.exp(-1j * np.pi * n * (n + 1) / (TxRxBase.SIG_LENGTH))

def showPlots():
    for scheme_index in range(NUM_SCHEMES + 1):
        plt.figure()
        extent = [-45, 45, -45, 45]  # [x_min, x_max, y_min, y_max]
        plt.imshow(np.abs(rx_samples[:, :, scheme_index, :]), aspect='auto', cmap='viridis', origin='lower', 
                    extent=extent, vmin=0.0, vmax=1.0)
        # Set tick marks to match usedAngles
        xticks = TxRxBase.usedAngles
        xlabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
        plt.xticks(xticks, xlabels)
        ylabels = [str(angle) if i % 2 == 0 else '' for i, angle in enumerate(xticks)]
        plt.yticks(xticks, ylabels)
        plt.colorbar(label='RSSI') # Add a color legend
        plt.title(f"Beamforming Heatmap (Scheme {scheme_index})")
        plt.xlabel("Rx Angle Index")
        plt.ylabel("Tx Angle Index")
        plt.tight_layout()
        plt.show()

while True:
    tx_angle_index = int(np.floor(angle_index / TxRxBase.NUM_ANGLES))
    rx_angle_index = angle_index % TxRxBase.NUM_ANGLES

    if scheme_index == 0: # If scheme index is 0, use Zadoff-Chu sequence
        TxRxBase.transmit(zc_sequence, TxRxBase.usedAngles[tx_angle_index], TxRxBase.usedAngles[rx_angle_index])
    else: # If scheme index is greater than 0, use PSK modulation
        sig = TxRxBase.psk_modulate(SCHEMES[scheme_index-1], None, rng)
        dataToSend = np.array(sig)
        dataToSend = TxRxBase.encode_data(dataToSend) # Convert to bytes
        sock4.sendto(dataToSend, ('localhost', GNU_UDP_TX_CONST_GUI_PORT))
        shaped = TxRxBase.pulseShapeTx(sig, filt)
        TxRxBase.transmit(shaped, TxRxBase.usedAngles[tx_angle_index], TxRxBase.usedAngles[rx_angle_index])

    angle_index = angle_index + 1
    if angle_index >= TxRxBase.NUM_ANGLES * TxRxBase.NUM_ANGLES: # If reached max angle index
        angle_index = 0 # Reset angle index
        scheme_index = scheme_index + 1
        if scheme_index >= NUM_SCHEMES + 1: # If reached max scheme index
            save_dict = {
                'rx_samples': rx_samples,
                'rx_samples_downsampled': rx_samples_downsampled,
                'seed': seed
            }
            np.save('C:/Users/pwfau/OneDrive/Documents/GNURadio/Scripts/rx_samples.npy', save_dict)
            exit(0) # Exit the program