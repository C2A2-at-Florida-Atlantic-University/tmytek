import numpy as np
import matplotlib.pyplot as plt
import socket
import random
import time
import TxRxBase

GNU_UDP_RX_CONST_GUI_PORT = 5006 # Port for the Rx GUI to receive constalltion data
GNU_UDP_TX_CONST_GUI_PORT = 5008 # Port for the Tx GUI to receive constalltion data

TxRxBase.SAMPLE_RATE = 1000000 # Sample rate in Hz
TxRxBase.setUsedAngles(30 + 1) # Number of angles in the index book
TxRxBase.SIG_LENGTH = 256 * 2 - 13 # Length of the signal in samples
TxRxBase.DELAY_BETWEEN_PACKETS_MS = 500 # MAX delay between packets in milliseconds
TxRxBase.TMYTEK_BF_SETTLE_TIME_MS = 1 # Settle time for TMYTEK beamforming in milliseconds

filt = TxRxBase.rrcFilter # Filter for pulse shaping

sock3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # GUI rx constellation sock
sock4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # GUI rx constellation sock

rng = np.random.default_rng(np.random.SeedSequence().entropy) # Random number generator for PSK modulation
latest_seed = None

M = 4

tx_symz = np.array([i // (97 // 4) for i in range(97)])
tx_symz = np.random.randint(0, M, size=97)

# Function that will be called when a packet is received properly
def rx_callback(sig):
  global latest_seed, M, tx_symz
  sig = sig[13:] # Remove the Barker code
  sig = TxRxBase.pulseShapeRx(sig, filt)
  sig = TxRxBase.psk_correct(sig, M)
  dataToSend = np.array(sig)
  dataToSend = TxRxBase.encode_data(dataToSend) # Convert to bytes
  sock3.sendto(dataToSend, ('localhost', GNU_UDP_RX_CONST_GUI_PORT))
  rx_syms = TxRxBase.psk_demodulate(sig, M) # Demodulate the received signal
  #print(f"Detected symbols: {rx_syms}") # Print detected symbols
  rx_syms = rx_syms[TxRxBase.NUM_PILOTS:]
  rx_syms = rx_syms[:len(tx_symz)] # Truncate to the length of tx_symz
  ser = TxRxBase.get_ser(M, rx_syms, tx_symz) # Calculate symbol error rate
  print(f"SER: {ser*100:.0f}%") # Print the symbol error rate


# Start the UDP receiver for the GNU Radio sink
TxRxBase.start_rx_udp(rx_callback)
TxRxBase.Beamformer.setAngle(b't', 0)
TxRxBase.Beamformer.setAngle(b'r', 0)

time.sleep(0.5) # Wait for the beamformer to settle

while True:
  latest_seed = rng.integers(0, 2**32, dtype=np.uint32)
  tx_symz = np.random.randint(0, M, size=97)
  sig = TxRxBase.psk_modulate(M, tx_symz)
  dataToSend = np.array(sig)
  dataToSend = TxRxBase.encode_data(dataToSend) # Convert to bytes
  sock4.sendto(dataToSend, ('localhost', GNU_UDP_TX_CONST_GUI_PORT))
  shaped = TxRxBase.pulseShapeTx(sig, filt)
  TxRxBase.transmit(shaped)
  time.sleep(0.03) # Wait for the beamformer to settle