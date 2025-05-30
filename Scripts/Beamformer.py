import subprocess
import socket
import time

BF_PORT = 5050

# Returns true if the beamforming server is up
def is_bf_server_up():
  try:
    with socket.create_connection(("localhost", BF_PORT), timeout=1) as _:
      return True
  except:
    return False

# Start the service if its not already running
def initService():
  # Start bf background server if needed
  if not is_bf_server_up():
    print("Starting beamforming controller server...")
    subprocess.Popen(
      ["python", "C:/Users/pwfau/OneDrive/Documents/GNURadio/Scripts/BeamformerService.py"],
      creationflags=subprocess.CREATE_NEW_CONSOLE  # This detaches it on Windows
    )
    for _ in range(10):
      time.sleep(1)
      if is_bf_server_up():
        break
    else:
      raise RuntimeError("Controller server didn't start in time!")
  else:
    print("Beamforming server already running")


# Kill the process
def killService():
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("localhost", BF_PORT))
        s.sendall(b'q\x00')
  except Exception as e:
    print("Bf service not running, can't kill it")
    return

# Set the angle 
def setAngle(mode, angle):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("localhost", BF_PORT))
        angle_byte = int(angle + 45).to_bytes(1, 'big')
        s.sendall(mode + angle_byte)
        #print(f"Sent command: mode={mode.decode()}, angle={angle}")