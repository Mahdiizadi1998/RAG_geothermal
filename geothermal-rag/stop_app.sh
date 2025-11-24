#!/bin/bash
# Script to stop existing Gradio instance on port 7860

echo "========================================"
echo "Stopping Gradio on Port 7860"
echo "========================================"
echo

echo "Checking for processes using port 7860..."
PID=$(lsof -ti:7860)

if [ -n "$PID" ]; then
    echo "Found process with PID: $PID"
    echo "Attempting to terminate..."
    kill -9 $PID
    
    if [ $? -eq 0 ]; then
        echo
        echo "========================================"
        echo "Successfully stopped process"
        echo "========================================"
        echo
        echo "You can now run: python app.py"
    else
        echo
        echo "========================================"
        echo "Failed to stop process"
        echo "========================================"
        echo "Try running with sudo: sudo ./stop_app.sh"
    fi
else
    echo
    echo "========================================"
    echo "No process found using port 7860"
    echo "========================================"
    echo "Port is already available"
fi

echo
