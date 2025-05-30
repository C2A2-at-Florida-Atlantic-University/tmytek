import socket
import threading
import pickle
import time
import sys
import os

with open('C:/Users/pwfau/OneDrive/Documents/GNURadio/Scripts/Beamformer.pid', 'w') as f:
  f.write(str(os.getpid()))

directory_path = 'C:/Users/pwfau/OneDrive/Documents/MATLAB/tmytek/BeamformingTest/controller.py'
sys.path.append(os.path.dirname(directory_path))
import controller

BF_PORT = 5050

print("Starting controller server...")
controller.initDevices()
print("Devices initialized!")

# Function to handle incoming client connections
def handle_client(conn):
  try:
    data = conn.recv(2)
    if len(data) != 2:
      print("Invalid data received")
      return
    if data == b'q\x00':
      print("Shutdown requested!")
      os._exit(0)
    
    mode = chr(data[0])
    angle = int(data[1]) - 45
    if angle < -45 or angle > 45:
      print("Invalid angle received")
      return

    #print(f"Received mode={mode}, angle={angle}")
    
    if mode == 't':
      controller.setBeamAngle(controller.txBBox, angle)
    elif mode == 'r':
      controller.setBeamAngle(controller.rxBBox, angle)
    else:
      print(f"Unknown mode: {mode}")
  except Exception as e:
    print("Error handling client:", e)

# Main loop to accept incoming connections
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
  #s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  s.bind(("localhost", BF_PORT))
  s.listen()
  print(f"Controller server listening on {BF_PORT}")
  
  while True:
    conn, _ = s.accept()
    threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

print("Beamformer controller server stopped.")