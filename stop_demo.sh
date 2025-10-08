#!/bin/bash

echo "🛑 Stopping TrackMania RL Demo..."
echo "==============================="

# Kill all demo processes
pkill -f "learning_dashboard.py" && echo "✅ Stopped Learning Dashboard"
pkill -f "trackmania_viewer.py" && echo "✅ Stopped Racing Viewer"
pkill -f "environment_server.py" && echo "✅ Stopped Environment Server"
pkill -f "model_server.py" && echo "✅ Stopped Model Server"

# Clean up log files
rm -f dashboard.log viewer.log environment.log model.log

echo ""
echo "🧹 All demo processes stopped and logs cleaned up."
echo "Ready for next demo session!"