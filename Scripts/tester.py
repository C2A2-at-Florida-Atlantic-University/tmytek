import numpy as np
import matplotlib.pyplot as plt
import time
import TxRxBase

TxRxBase.SAMPLE_RATE = 1000000 # Sample rate in Hz
TxRxBase.setUsedAngles(30 + 1) # Number of angles in the index book
print(f"Used angles: {TxRxBase.usedAngles}")
TxRxBase.SIG_LENGTH = 256 * 2 - 13 # Length of the signal in samples
TxRxBase.DELAY_BETWEEN_PACKETS_MS = 100 # MAX delay between packets in milliseconds
TxRxBase.TMYTEK_BF_SETTLE_TIME_MS = 1 # Settle time for TMYTEK beamforming in milliseconds

# Function that will be called when a packet is received properly
def rx_callback(sig):
    pass

# Start the UDP receiver for the GNU Radio sink
TxRxBase.start_rx_udp(rx_callback)
TxRxBase.Beamformer.setAngle(b't', 0)
TxRxBase.Beamformer.setAngle(b'r', 0)

time.sleep(0.5) # Wait for the beamformer to settle

# Generate time vector
t = np.arange(TxRxBase.SIG_LENGTH) / TxRxBase.SAMPLE_RATE
# Generate Zadoff-Chu sequence
n = np.arange(TxRxBase.SIG_LENGTH)
zc_sequence = np.exp(-1j * np.pi * n * (n + 1) / (TxRxBase.SIG_LENGTH))

while True:
    TxRxBase.transmit(zc_sequence)