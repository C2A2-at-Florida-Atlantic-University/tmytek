import subprocess
import socket
import time
import threading
from collections import deque
import struct
import numpy as np
import os
import sys
from sk_dsp_comm.sigsys import sqrt_rc_imp
from scipy.stats import norm
import matplotlib.pyplot as plt
import Beamformer

# Function to decode the received data (similar to MATLAB code)
def decode_data(data):
    # Convert the byte data to floats (single precision)
    rawData = np.frombuffer(data, dtype=np.uint8)  # Read as raw uint8 data
    complexData = rawData.view(np.float32) # Typecast to single precision float32
    # Separate real and imaginary parts (assuming interleaved format)
    realPart = complexData[::2] # Real parts at odd indices
    imagPart = complexData[1::2] # Imaginary parts at even indices
    # Combine real and imaginary parts into complex numbers
    receivedSignal = realPart + 1j * imagPart
    return receivedSignal

# Function to encode the data to be transmitted (similar to MATLAB code)
def encode_data(tx_samples):
    # Step 1: Allocate space for interleaved data (2 * length of txSig for real and imaginary parts)
    interleavedData = np.zeros(2 * len(tx_samples), dtype=np.float32)
    # Step 2: Interleave real and imaginary parts
    interleavedData[0::2] = np.real(tx_samples)  # Real parts at odd indices (0, 2, 4, ...)
    interleavedData[1::2] = np.imag(tx_samples)  # Imaginary parts at even indices (1, 3, 5, ...)
    # Step 3: Typecast to uint8 (8-bit unsigned integers)
    dataToSend = interleavedData.astype(np.float32).tobytes()
    return dataToSend

# Function to set the used angles
def setUsedAngles(num_angles):
    global usedAngles, NUM_ANGLES
    NUM_ANGLES = num_angles
    usedAngles = np.floor(np.linspace(-45, 45, NUM_ANGLES)).astype(int)

# Function to generate the raised cosine filter (similar to MATLAB code)
def rcosdesign(beta, sps, span):
    N = span * sps + 1 # Ensure odd length for symmetry
    t = np.arange(-N//2, N//2 + 1) / sps
    h = np.zeros_like(t) # Initialize filter
    eps = 1e-8 # Small epsilon to avoid division by zero
    idx_normal = np.abs(1 - (2 * beta * t)**2) > eps # Indices where denominator is NOT zero
    h[idx_normal] = ( # Raised Cosine formula
        np.sinc(t[idx_normal]) * np.cos(np.pi * beta * t[idx_normal]) /
        (1 - (2 * beta * t[idx_normal])**2)
    )
    # Handle singularities using l'Hôpital's rule
    idx_singular = np.where(np.abs(1 - (2 * beta * t)**2) <= eps)[0]
    for i in idx_singular:
        h[i] = (np.pi / 4) * np.sinc(1 / (2 * beta))
    h /= np.sqrt(np.sum(h**2)) # Normalize energy
    return h, t

# Function to generate the root raised cosine filter (similar to MATLAB code)
def rrcosdesign(beta, sps, span):
    N = span * sps + 1 # Must be odd for symmetric filter
    t = np.arange(-np.floor(N/2), np.floor(N/2)+1) / sps # Time vector centered at 0
    t += 10e-8 # Avoid division by zero at t=0
    # Root Raised Cosine Filter formula
    h = (np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))) / (np.pi * t * (1 - (4 * beta * t) ** 2))
    h /= np.sqrt(np.sum(h**2)) # Normalize filter energy
    return h, t

# Basically ppf but more accurate
def max_gaussian_ppf(N, P_FA):
  logN = np.log(N)
  sqrt_2logN = np.sqrt(2 * logN)
  correction = (np.log(np.log(N)) + np.log(4 * np.pi)) / (2 * sqrt_2logN)
  quantile_adjustment = -np.log(-np.log(1 - P_FA)) / sqrt_2logN
  return sqrt_2logN - correction + quantile_adjustment

# Function to upsample then pulse shape the signal (similar to MATLAB code)
def pulseShapeTx(sig, filter):
    # Upsample
    upsampled = np.zeros(len(sig) * SPS, dtype=complex)
    upsampled[::SPS] = sig # Insert symbol, then SPS-1 zeros
    # Apply filter
    filtered = np.convolve(upsampled, filter, mode='same')
    return filtered

# Function to downsample then pulse shape the signal (similar to MATLAB code)
# NOTE this assumes that the signal starts right after the barker code was detected (between 0 and SPS-1 samples)
def pulseShapeRx(sig, filter):
    # Apply filter
    sig = np.convolve(sig, filter, mode='same')
    # Timing recovery: estimate optimal sampling phase
    energy_per_phase = np.zeros(SPS, dtype=float)
    for m in range(SPS):
        phase_samples = sig[m::SPS]  # take every sps-th sample starting from m
        energy_per_phase[m] = np.sum(np.abs(phase_samples)**2)
    optimal_phase = np.argmax(energy_per_phase) # Find the best phase
    # Phase align and downsample
    sig = sig[optimal_phase:] # Align to the optimal phase
    sig = sig[:len(sig) - (len(sig) % SPS)] # Ensure length is a multiple of SPS
    downsampled = sig[::SPS] # Downsample by taking every SPS-th sample
    return downsampled

# Funciton to do PSK modulation
def psk_modulate(M, syms=None, seed=None):
    k = int((SIG_LENGTH-13) / SPS - NUM_PILOTS) # Number of symbols to transmit
    if syms is None: # If no symbols are provided, generate random symbols
        if seed is None:
            seed = np.random.SeedSequence().entropy
        rng = np.random.default_rng(seed) # Random number generator for PSK modulation
        syms = rng.integers(0, M, size=k) # Generate random symbols
    pilots = np.array([i % M for i in range(NUM_PILOTS)]) # Generate pilot symbols
    syms = np.concatenate((pilots, syms)) # Concatenate symbols and pilots
    #print(f"Transmitted symbols: {syms}") # Print transmitted symbols
    symbols = np.exp(1j * (2 * np.pi * syms / M))
    return symbols

# Function to do PSK phase and magnitude correction
def psk_correct(sig, M):
    pilots = np.array([i % M for i in range(NUM_PILOTS)]) # Generate pilot symbols
    expected_pilots = np.exp(1j * (2 * np.pi * pilots / M)) # Expected pilot symbols
    # Estimate channel: complex gain = magnitude + phase
    # Avoid averaging phases directly — average the ratio
    channel_estimates = sig[0:NUM_PILOTS] / expected_pilots # Estimate channel using pilots
    # Phase-only correction
    avg_phase = np.angle(np.mean(channel_estimates))  # Just the angle
    #print(f"Average phase offset:{np.rad2deg(avg_phase)}°") # Print average phase
    # Apply inverse of estimated channel
    rx_phase_corrected = sig * np.exp(-1j * avg_phase)
    rx_normalized = rx_phase_corrected / np.sqrt(np.mean(np.abs(rx_phase_corrected)**2)) # Normalize energy
    return rx_normalized

# Function to PSK demodulate a phase-corrected signal
def psk_demodulate(sig, M):
    #sig = sig[NUM_PILOTS:] # Remove pilots
    constellation = np.exp(1j * 2 * np.pi * np.arange(M) / M) # Generate constellation points
    #plt.plot(np.real(sig),np.imag(sig), 'o') # Plot constellation points
    #plt.title(f"Constellation for M={M}")
    distances = np.abs(sig[:, np.newaxis] - constellation[np.newaxis, :]) # Calculate distances to constellation points
    detected_syms = np.argmin(distances, axis=1) # Find closest constellation point
    return detected_syms

# Function to calculate the error vector magnitude (EVM)
def get_evm(rx_const, tx_const):
    evm = np.sqrt(np.mean(np.real(rx_const) - np.real(tx_const))**2 + np.mean(np.imag(rx_const) - np.imag(tx_const))**2) / np.sqrt(np.mean(np.real(tx_const)**2 + np.imag(tx_const))**2)
    return evm

# Function to get the symbol error rate (SER)
def get_ser(M, rx_syms, tx_syms=None, seed=None):
    if tx_syms is None: # If no transmitted symbols are provided, generate random symbols
        rng = np.random.default_rng(seed) # Random number generator for PSK modulation
        tx_syms = rng.integers(0, M, size=len(rx_syms)) # Generate random symbols using the same seed
    # Calculate the number of symbol errors
    num_errors = 0
    for i in range(min(len(rx_syms), len(tx_syms))): # Iterate over the symbols
        if rx_syms[i] != tx_syms[i]: # If the received symbol does not match the transmitted symbol
            num_errors += 1
    # Calculate the symbol error rate (SER)
    ser = num_errors / len(rx_syms) # SER = number of errors / total symbols
    return ser  

# Function to qam modulate a signal
def qam_modulate(M, syms=None, seed=None):
  k = int((SIG_LENGTH-13) / SPS - NUM_PILOTS)
  if syms is None:
    if seed is None:
      seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(seed)
    syms = rng.integers(0, M, size=k)
  pilots = np.array([i % M for i in range(NUM_PILOTS)])
  syms = np.concatenate((pilots, syms))
  m_side = int(np.sqrt(M))
  if m_side**2 != M:
    raise ValueError("M must be a perfect square for square QAM.")
  real = 2 * (syms % m_side) - m_side + 1
  imag = 2 * (syms // m_side) - m_side + 1
  symbols = real + 1j * imag
  # Normalize average power to 1
  symbols /= np.sqrt(np.mean(np.abs(symbols)**2))
  return symbols

# Function to correct the QAM signal for phase and magnitude
def qam_correct(sig, M):
  m_side = int(np.sqrt(M))
  pilots = np.array([i % M for i in range(NUM_PILOTS)])
  real = 2 * (pilots % m_side) - m_side + 1
  imag = 2 * (pilots // m_side) - m_side + 1
  expected_pilots = real + 1j * imag
  expected_pilots /= np.sqrt(np.mean(np.abs(expected_pilots)**2))
  channel_estimates = sig[0:NUM_PILOTS] / expected_pilots
  h = np.mean(channel_estimates)
  corrected = sig / h
  return corrected

# Function to calculate the symbol error rate (SER) for QAM modulation
def qam_demodulate(sig, M):
  m_side = int(np.sqrt(M))
  real_levels = np.arange(-m_side+1, m_side+1, 2)
  imag_levels = np.arange(-m_side+1, m_side+1, 2)
  constellation = np.array([x + 1j*y for y in imag_levels for x in real_levels])
  constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
  distances = np.abs(sig[:, np.newaxis] - constellation[np.newaxis, :])
  detected_syms = np.argmin(distances, axis=1)
  return detected_syms


GNU_UDP_SINK_PORT = 5000 # Port number that matches the UDP Sink in GNU Radio
GNU_UDP_TX_SIG_GUI_PORT = 5010 # Port number that matches the UDP Rx GUI Source in GNU Radio
GNU_UDP_RX_SIG_GUI_PORT = 5004 # Port number that matches the UDP Rx GUI Source in GNU Radio
GNU_UDP_SOURCE_PORT = 5002 # Port number that matches the UDP Source in GNU Radio

SAMPLE_RATE = 1000000 # Sample rate in Hz
SIG_LENGTH = 256 * 2 - 13 # Length of the signal in samples
NUM_ANGLES = 30 + 1 # Number of angles in the index book

RX_UDP_BUFFER_SIZE = 1024 # Number of bytes at a time that the UDP sink will send
P_FA = 0.001 # False alarm probability for Barker code detection
STATUS_CHECK_INTERVAL = 1 # Interval to check the status of the system in seconds

TMYTEK_BF_SETTLE_TIME_MS = 1 # Time to wait for the beam to settle in ms
DELAY_BETWEEN_PACKETS_MS = 2 # MAX delay between packets in ms
TX_REPETITIONS = 1 # Number of times to send the packet
MIN_SNR = 10 # Min expected SNR in dB for Barker code detection
MAX_PACKET_TRIES = 3 # Maximum number of tries to send a packet
NUM_NOISE_PACKETS_TO_RECORD = 5 # Number of rx noise packets to record 

NUM_PILOTS = 24 # Number of pilots in the signal

SPS = 4 # Samples per symbol
FILT_LEN = 64 # Filter length in symbols
ROLLOFF = 0.25 # Roll-off factor for the filter
rcFilter, t_h = rcosdesign(ROLLOFF, SPS, FILT_LEN) # Generate the filter
rrcFilter, t_h = rrcosdesign(ROLLOFF, SPS, FILT_LEN) # Generate the filter

# Barker-13 sequence
barker_code = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1])

usedAngles = None
setUsedAngles(NUM_ANGLES)

# Global variables
tx_sock = None # Socket for transmitting
tx_gui_sock = None # Socket for transmitting to GUI
got_packet = threading.Event() # Flag to indicate if a packet has been received
expect_packet = threading.Event() # Flag to indicate if a packet is expected to be received
got_packet.set() # Set the flag to indicate that no packet has been received yet
record_noise_packets = threading.Event() # Flag to indicate if noise packets should be recorded
received_pacekts = 0
missed_packets = 0
auto_noise_iter = 0
avg_noise_power = 0 # Average noise power (set dynamically)


#Beamformer.killService() # Kill any existing service
#exit(0)

Beamformer.initService()

# Function to start the UDP receiver (will be non-blocking)
def rx_udp(callback_func):
    global got_packet, received_pacekts, missed_packets, avg_noise_power
    print("Starting UDP receiver...\n")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create a UDP socket from the GNU Radio sink
    sock.bind(("localhost", GNU_UDP_SINK_PORT)) # Bind to the UDP sink port
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # GUI rx signal sock

    threshold = 0.5 # Default threshold for Barker code detection

    # Holds energy of last few packets
    signal_strengths = deque(maxlen=10)

    sample_count = 0 # Counter for number of samples
    start_time = time.time() # Track the time
    buffer = np.zeros(SIG_LENGTH + 13, dtype=np.complex64) # Holds the captured signal
    buffer_cursor = 0 # Cursor for the buffer
    capturing = False # Flag to indicate if capturing is in progress 

    while True:
        # Receive a message from the UDP socket
        data, addr = sock.recvfrom(RX_UDP_BUFFER_SIZE)
        sample_count += len(data) / 8 # Increment by the size of the data received
        # Print the number of samples received every second
        if time.time() - start_time >= STATUS_CHECK_INTERVAL:
            sample_percent = int(100 * sample_count / (SAMPLE_RATE * STATUS_CHECK_INTERVAL)) # Calculate percentage of samples received
            print(f"Samples in last {STATUS_CHECK_INTERVAL}s: {sample_percent:d}%, Recieived: {received_pacekts}, Missed: {missed_packets}")
            if len(signal_strengths) > 0:
                snr = 10 * np.log10((np.mean(signal_strengths)-avg_noise_power) / avg_noise_power) # Calculate SNR
                print(f"Avg Signal Power: {np.mean(signal_strengths)-avg_noise_power:.8f}, Avg Noise Pwr: {avg_noise_power:.8f}, Avg SNR: {snr:.1f}dB, Threshold: {threshold:.8f}")
            # Reset counters
            start_time = time.time()
            sample_count = 0
            received_pacekts = 0
            missed_packets = 0

        # Dynamically adjust the threshold based on the noise power now
        if record_noise_packets.is_set(): # There should not be any transmission rn
            avg_noise_power = np.mean(np.abs(decode_data(data))**2) # Calculate average noise power
            if avg_noise_power > 0.005:
                plt.plot(np.real(decode_data(data)), label='Noise')
                plt.show()
            min_sig_power = avg_noise_power * 10**(MIN_SNR/10) # Calculate minimum signal power
            threshold = np.sqrt(13*avg_noise_power)*max_gaussian_ppf(SIG_LENGTH-13, P_FA) + min_sig_power
            record_noise_packets.clear() # Clear the flag to stop recording noise packets
            continue # Skip the rest of the loop

        if not expect_packet.is_set(): # If the TX is not waiting for a packet to be received, dont do anything
            continue

        new_samples = decode_data(data) # Convert raw bytes to complex samples
        new_samples_cursor = 0
        while new_samples_cursor < len(new_samples): # While there are samples to process
            num_to_copy = min(len(new_samples) - new_samples_cursor, len(buffer) - buffer_cursor) # Number of samples to copy
            buffer[buffer_cursor:buffer_cursor+num_to_copy] = new_samples[new_samples_cursor:new_samples_cursor+num_to_copy] # Copy new samples to buffer
            buffer_cursor += num_to_copy # Update buffer cursor
            new_samples_cursor += num_to_copy # Update new samples cursor
            if buffer_cursor < len(buffer): # If the buffer is not full (there are also not new samples left to fill it)
                continue # Exit and wait for more samples
            
            if not capturing and not got_packet.is_set(): # Activley looking for Barker code
                correlation = np.abs(np.correlate(buffer, barker_code, mode='valid')) # Compute cross-correlation on the buffer
                detected_indices = np.where(correlation > 2*threshold)[0] # Find indices where correlation exceeds threshold
                if detected_indices.size > 0: # Barker (OR POTENTAILLY BARKER SIDELOBE) detected
                    best_idx = np.argmax(correlation) # Find the index of the best correlation
                    buffer_cursor = len(buffer) - best_idx # Update buffer cursor so the contets start from the detected index
                    buffer[0:buffer_cursor] = buffer[best_idx:] # Shift the buffer to start from the detected index
                    #plt.plot(np.real(new_samples))
                    #plt.axvline(x=best_idx, color='r', linestyle='--')
                    #plt.show()
                    capturing = True # Start capturing now (and potentially finish if the buffer is full)
                else: # Barker not detected
                    buffer_cursor = 0 
                    #buffer_cursor = 13 # Reset the buffer cursor to start with the tail of the current buffer
                    #buffer[0:buffer_cursor] = buffer[-13:] # Keep the last 13 samples in the buffer
            if capturing and buffer_cursor == len(buffer): # If capturing just finished and the buffer is full
                capturing = False # Stop capturing
                sig_strength = np.mean(np.abs(buffer[13:])**2) # Calculate average signal strength
                got_packet.set() # Set the flag to indicate that a packet has been received for tx
                expect_packet.clear() # Set the flag to indicate that a packet is not expected to be received any more
                callback_func(buffer) # Call the callback function with the captured buffer
                signal_strengths.append(sig_strength) # Calculate average signal strength and append to the list
                dataToSend = encode_data(buffer) # Convert to bytes so the GUI can read it
                sock2.sendto(dataToSend, ('localhost', GNU_UDP_RX_SIG_GUI_PORT)) # Send to GUI
                buffer_cursor = 0 # Reset the buffer cursor to start from the beginning of the buffer
                break # We don't need to process any of the remaining new samples since we have our signal
        

# Function to start the rx udp receiver in its own thread
def start_rx_udp(callback_func):
    rx_thread = threading.Thread(target=rx_udp, args=(callback_func,), daemon=True)
    rx_thread.start()

# Function to transmit the samples (blocking)
def transmit(tx_samples, rx_angle=None, tx_angle=None):
    global received_pacekts, missed_packets, tx_sock, tx_gui_sock, auto_noise_iter

    if auto_noise_iter == 0: # Every x iterations, tell rx to record noise packets where there is no transmission
        time.sleep(0.03) # Wait for a bit to ensure no transmission is happenings
        record_noise_packets.set() # Set the flag to record noise packets
        time.sleep(0.015) # Wait for a bit to ensure no transmission is happening
        record_noise_packets.clear() # Clear the flag to stop recording noise packets
    auto_noise_iter = auto_noise_iter + 1 if auto_noise_iter < 100 else 0 # Increment the noise packet counter

    if tx_sock == None:
        print("Starting UDP transmitter...\n")
        tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create UDP socket for transmission
        tx_gui_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # GUI tx signal sock

    empty_samples = np.zeros((SIG_LENGTH - len(tx_samples)), dtype=complex) # Create empty samples for padding
    if len(tx_samples) < SIG_LENGTH + 13: # If the samples are too long, truncate them
        tx_samples = np.concatenate((tx_samples.astype(complex), empty_samples.astype(complex))) # Pad the samples with zeros
    tx_samples = np.concatenate((barker_code.astype(complex), tx_samples.astype(complex))) # Prepend Barker code to the samples
    dataToSend = encode_data(tx_samples) # Convert to bytes
    fillerToSend = encode_data(np.zeros(13 + SIG_LENGTH, dtype=complex)) # Create filler data to send to the GUI

    if tx_angle is not None: # If tx angle is set
        Beamformer.setAngle(b't', tx_angle)
    if rx_angle is not None: # If rx angle is set
        Beamformer.setAngle(b'r', rx_angle)
    expect_packet.set() # Set the flag to indicate that a packet is expected to be received
    time.sleep(TMYTEK_BF_SETTLE_TIME_MS / 1000.0) # Wait for the beams to settle
    tx_gui_sock.sendto(dataToSend, ('localhost', GNU_UDP_TX_SIG_GUI_PORT)) # Send to GUI
    for packet_tries in range(MAX_PACKET_TRIES): # Try to send the packet and wait for reception multiple times
        tx_sock.sendto(fillerToSend, ('localhost', GNU_UDP_SOURCE_PORT)) # Send filler data to the GUI
        for i in range(TX_REPETITIONS): # Send packet (send multiple times to ensure reception)
            tx_sock.sendto(dataToSend, ('localhost', GNU_UDP_SOURCE_PORT))
        got_packet.wait(timeout=DELAY_BETWEEN_PACKETS_MS / 1000.0) # Wait for the got_packet to be set (timeout is the delay between packets)

        if got_packet.is_set(): # If packet received
            received_pacekts += 1 # Increment received packets
            break # Break out of the loop if packet is received
        else: # If packet not received
            missed_packets += 1 # Increment missed packets

    did_get_packet = got_packet.is_set() # Store the received packet flag
    if not did_get_packet: # If packet not received
        print("Packet not received after max tries")
    got_packet.clear() # Reset packet received flag 
    #tx_sock.sendto(fillerToSend, ('localhost', GNU_UDP_SOURCE_PORT)) # Send filler data to the GUI
    return did_get_packet # Return the received packet flag
