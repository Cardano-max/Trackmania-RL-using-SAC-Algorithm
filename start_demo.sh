#!/bin/bash

# TrackMania RL Demo Startup Script
# This script prepares everything for your video demo

echo "🏁 TrackMania RL System - Demo Setup"
echo "======================================"
echo ""

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)
echo "📂 Project directory: $PROJECT_DIR"
echo ""

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Stop any running systems
echo "🛑 Stopping any existing systems..."
lsof -ti:7001,8080,8081 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 2
echo "✅ Ports cleared"
echo ""

# Start the synchronized demo system
echo "🚀 Starting Synchronized RL + 3D System..."
python3 synchronized_rl_3d_system.py &
DEMO_PID=$!
echo "✅ Demo system starting (PID: $DEMO_PID)"
echo ""

# Wait for system to initialize
echo "⏳ Waiting for system to initialize..."
sleep 5

# Start the simulation
echo "🏎️ Starting RL simulation..."
curl -s -X POST http://localhost:7001/api/simulation/start > /dev/null
echo "✅ Simulation started"
echo ""

echo "======================================"
echo "✅ DEMO SYSTEM READY!"
echo "======================================"
echo ""
echo "🌐 Open in browser: http://localhost:7001"
echo ""
echo "📊 System Status:"
curl -s http://localhost:7001/api/status | jq '.simulation_active, .cars | length' 2>/dev/null || echo "Checking..."
echo ""
echo "📋 For two-container demo, run:"
echo "   Terminal 1: cd environment && python3 environment_server.py"
echo "   Terminal 2: cd model && python3 model_server.py"
echo ""
echo "🎬 Ready for video recording!"
echo "📄 See VIDEO_DEMO_SCRIPT.md for full script"
echo ""
echo "Press Ctrl+C to stop when done."
echo ""

# Keep script running
wait $DEMO_PID
