#!/bin/bash

echo "🏁 TrackMania RL - Client Demo Launcher"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "learning_dashboard.py" 2>/dev/null
pkill -f "trackmania_viewer.py" 2>/dev/null
pkill -f "environment_server.py" 2>/dev/null
sleep 2

# Activate virtual environment
source venv/bin/activate

echo "🚀 Starting TrackMania RL Demo Components..."
echo ""

# Start the learning dashboard in background
echo "📊 Starting Learning Dashboard (http://localhost:4000)..."
python3 tmrl_enhanced/learning_dashboard.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Wait for dashboard to start
sleep 3

# Check if dashboard started successfully
if ps -p $DASHBOARD_PID > /dev/null; then
    echo "✅ Learning Dashboard started successfully (PID: $DASHBOARD_PID)"
else
    echo "❌ Learning Dashboard failed to start. Check dashboard.log"
    exit 1
fi

echo ""
echo "🎯 Client Demo Ready!"
echo "==================="
echo ""
echo "📋 Demo Options:"
echo ""
echo "1. 🧠 Interactive RL Learning Dashboard"
echo "   URL: http://localhost:4000"
echo "   Action: Click '🚀 Start Training' to see live learning"
echo ""
echo "2. ⚡ Quick Command Line Demo"
echo "   Command: python3 simple_rl_demo.py"
echo "   Shows: 200 episodes of Q-Learning in 30 seconds"
echo ""
echo "3. 🎬 Racing Visualization (Alternative)"
echo "   Command: python3 viewer/trackmania_viewer.py"
echo "   URL: http://localhost:3000"
echo ""
echo "💡 Recommended: Start with Option 1 for best client presentation"
echo ""
echo "🔍 To check status: ./check_logs.sh"
echo "🛑 To stop demo: ./stop_demo.sh"
echo ""

# Try to open browser automatically (macOS)
if command -v open >/dev/null 2>&1; then
    echo "🌐 Opening browser..."
    open http://localhost:4000
fi

echo "✅ Demo is running! Press Ctrl+C to stop."

# Keep script running and show live status
while true; do
    sleep 5
    if ! ps -p $DASHBOARD_PID > /dev/null; then
        echo "⚠️  Dashboard process stopped unexpectedly"
        break
    fi
done