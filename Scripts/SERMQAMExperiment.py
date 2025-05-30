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

M = 16

tx_symz = np.random.randint(0, M, size=100)
tx_const = None


nBits = 0.1e6
numSteps = 20
pwrs = np.logspace(np.log10(0.01), np.log10(1), numSteps)
print(f"Power levels: {pwrs}")
snrs = np.zeros(numSteps) # Initialize the SNR array
sers = np.zeros(numSteps) # Initialize the SER array
evms = np.zeros(numSteps) # Initialize the EVM array
cStep = 0 # Current step index
tests = 0

# Function that will be called when a packet is received properly
def rx_callback(sig):
  global latest_seed, M, tx_symz, tx_const, serz, tests
  #plt.plot(sig)
  #plt.show()
  sig = sig[13:] # Remove the Barker code
  sig = TxRxBase.pulseShapeRx(sig, filt)
  noise_power = TxRxBase.avg_noise_power * TxRxBase.SPS
  try:
    snr = 10 * np.log10((np.mean(np.abs(sig)**2) - noise_power) / noise_power) # Calculate SNR
  except ZeroDivisionError: 
    print("ZeroDivisionError: avg_noise_power is zero")
    snr = 0
  sig = TxRxBase.qam_correct(sig, M)
  evm = TxRxBase.get_evm(sig, tx_const) # Calculate the error vector magnitude
  dataToSend = np.array(sig)
  dataToSend = TxRxBase.encode_data(dataToSend) # Convert to bytes
  sock3.sendto(dataToSend, ('localhost', GNU_UDP_RX_CONST_GUI_PORT))
  rx_syms = TxRxBase.qam_demodulate(sig, M) # Demodulate the received signal
  #print(f"Detected symbols: {rx_syms}") # Print detected symbols
  rx_syms = rx_syms[TxRxBase.NUM_PILOTS:]
  rx_syms = rx_syms[:len(tx_symz)] # Truncate to the length of tx_symz
  ser = TxRxBase.get_ser(M, rx_syms, tx_symz) # Calculate symbol error rate
  snrs[cStep] += snr # Store the SNR for the current step
  sers[cStep] += ser # Increment the SER for the current step
  evms[cStep] += evm # Increment the EVM for the current step
  tests += len(tx_symz) * np.log2(M) # Increment the number of tests
  print(f"Step: {cStep}, SER: {ser*100:.0f}%, EVM: {evm*100:.0f}%, Tests: {100*tests/nBits:.2f}%") # Print the symbol error rate


# Start the UDP receiver for the GNU Radio sink
TxRxBase.start_rx_udp(rx_callback)
TxRxBase.Beamformer.setAngle(b't', 0)
TxRxBase.Beamformer.setAngle(b'r', 0)

time.sleep(0.5) # Wait for the beamformer to settle


while True:
  tx_symz = np.random.randint(0, M, size=100)
  sig = TxRxBase.qam_modulate(M, tx_symz)
  tx_const = sig.copy()
  sig = sig * pwrs[cStep] # Scale the signal by the current power level
  dataToSend = np.array(sig)
  dataToSend = TxRxBase.encode_data(dataToSend) # Convert to bytes
  sock4.sendto(dataToSend, ('localhost', GNU_UDP_TX_CONST_GUI_PORT))
  shaped = TxRxBase.pulseShapeTx(sig, filt)
  TxRxBase.transmit(shaped)
  time.sleep(0.06)
  if tests >= nBits:
    snrs[cStep] /= (tests / (len(tx_symz) * np.log2(M)))
    sers[cStep] /= (tests / (len(tx_symz) * np.log2(M)))
    evms[cStep] /= (tests / (len(tx_symz) * np.log2(M)))
    print(f"Step {cStep}: SNR: {snrs[cStep]:.2f}dB, SER: {sers[cStep]*100:.2f}%, EVM: {evms[cStep]*100:.2f}%")
    cStep += 1
    tests = 0
    if cStep >= numSteps:
      break
  
for i in range(numSteps):
  print(f"Step {i}: SNR: {snrs[i]:.2f}dB, SER: {sers[i]*100:.2f}%, EVM: {evms[i]*100:.2f}%")
