#!/bin/bash

# Path to the PID file (same as in your Python script)
PID_FILE="C:/path/to/your/pid/file.pid"

# Check if the PID file exists
if [ -f "$PID_FILE" ]; then
  # Read the PID from the file
  PID=$(cat "$PID_FILE")

  # Check if the PID corresponds to a running process
  if kill -0 $PID 2>/dev/null; then
    # Kill the process
    kill $PID
    echo "Process $PID killed successfully."
  else
    echo "No running process with PID $PID found."
  fi

  # Optionally, delete the PID file after killing the process
  rm "$PID_FILE"
else
  echo "PID file not found."
fi
