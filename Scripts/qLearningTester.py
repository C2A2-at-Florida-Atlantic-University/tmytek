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
TxRxBase.BARKER_THRESHOLD = 0.2 # Threshold for Barker code detection
TxRxBase.DELAY_BETWEEN_PACKETS_MS = 10 # MAX delay between packets in milliseconds
TxRxBase.TMYTEK_BF_SETTLE_TIME_MS = 1 # Settle time for TMYTEK beamforming in milliseconds

filt = TxRxBase.rrcFilter # Filter for pulse shaping

# Q-Learning 
numActions = 3
learningRate = 0.5
discountFactor = 0.95
explorationRate = 0.4
# Initialize Q1 matrix
Q1 = np.zeros((TxRxBase.NUM_ANGLES, numActions)) # Q1 matrix for Q-learning
Q1[0, 0] = -100 # Don't move left at angle 1
Q1[TxRxBase.NUM_ANGLES-1, 2] = -100 # Don't move right at max angle (index numAngles - 1, column 3→ index 2)
rxAngleIndexQ1 = random.randint(0, TxRxBase.NUM_ANGLES - 1) # 0-based indexing
rxAngleIndexQP = 0
lastRSSI = 0
action = 0 # Action taken

lastPrint = 0 # Last print time

numIters = 1000 # Number of iterations
rssis = np.zeros(numIters) # Array to store RSSI values for plotting
angles = np.zeros(numIters) # Array to store angles for plotting
iterIndex = 0 # Index for RSSI array

sock3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # GUI rx constellation sock 
sock4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # GUI rx constellation sock 

txAngle = 6 # Transmit angle

# Function that will be called when a packet is received properly
def rx_callback(sig):
    global rxAngleIndexQ1, rxAngleIndexQP, lastRSSI, Q1, action
    global iterIndex, rssis, lastPrint
    if iterIndex >= numIters: # If reached max RSSI array size
        return # Ignore the packet
    sig = sig[13:] # Remove the Barker code
    sig = TxRxBase.pulseShapeRx(sig, filt)
    rssi = np.mean(np.abs(sig)**2) # Calculate RSSI
    dataToSend = np.array(sig)
    dataToSend = TxRxBase.encode_data(dataToSend) # Convert to bytes
    sock3.sendto(dataToSend, ('localhost', GNU_UDP_RX_CONST_GUI_PORT))
    # Q-learning update
    reward = (rssi - lastRSSI) * 10
    lastRSSI = rssi
    Q1[rxAngleIndexQ1, action - 1] += learningRate * (
        reward + discountFactor * np.max(Q1[rxAngleIndexQP, :]) - Q1[rxAngleIndexQ1, action - 1]
    )
    rxAngleIndexQ1 = rxAngleIndexQP
    # Update RSSI array for plotting
    rssis[iterIndex] = rssi
    iterIndex += 1
    if time.time() - lastPrint > 1: # Print every second
        print(f"RSSI: {rssi:.2f}, Angle: {rxAngleIndexQ1} ({TxRxBase.usedAngles[rxAngleIndexQ1]})")
        lastPrint = time.time()

# Start the UDP receiver for the GNU Radio sink
TxRxBase.start_rx_udp(rx_callback)
TxRxBase.Beamformer.setAngle(b't', txAngle)
TxRxBase.Beamformer.setAngle(b'r', TxRxBase.usedAngles[rxAngleIndexQ1])

while True:
    if iterIndex >= len(rssis): # If reached max RSSI array size
        print(Q1)
        print("Optimal policy:")
        optimal_policy = np.argmax(Q1, axis=1)  # Gives optimal action for each state
        for i, a in enumerate(optimal_policy):
            print(f"State {i}: Action {a}")
        plt.figure(1)
        plt.plot(rssis)
        plt.title('RSSIs')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.figure(2)
        plt.plot(angles)
        plt.title('Angles')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
        iterIndex = 0 # Reset iterIndex for next run

    # Epsilon-greedy action selection
    if random.random() < explorationRate:
        action = random.randint(1, numActions) # Random action
        while (action == 1 and rxAngleIndexQ1 == 0) or (action == 3 and rxAngleIndexQ1 == TxRxBase.NUM_ANGLES - 1):
            action = random.randint(1, numActions)
    else: # Greedy action selection
        action = int(np.argmax(Q1[rxAngleIndexQ1, :])) + 1
    # Adjust angle based on action
    rxAngleIndexQP = max(0, min(TxRxBase.NUM_ANGLES - 1, rxAngleIndexQ1 + (action - 2))) # action 2 = no move
    rxAngle = TxRxBase.usedAngles[rxAngleIndexQP]
    angles[iterIndex] = rxAngle # Store angle for plotting

    sig = TxRxBase.psk_modulate(4, None)
    dataToSend = np.array(sig)
    dataToSend = TxRxBase.encode_data(dataToSend) # Convert to bytes
    sock4.sendto(dataToSend, ('localhost', GNU_UDP_TX_CONST_GUI_PORT))
    shaped = TxRxBase.pulseShapeTx(sig, filt)
    TxRxBase.transmit(shaped, txAngle, rxAngle)