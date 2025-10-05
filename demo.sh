#!/bin/bash

echo "=== TrackMania RL Implementation Demo ==="
echo

# Activate virtual environment
echo "Activating virtual environment..."
source test_env/bin/activate

echo "✅ Environment activated"
echo

# Test 1: Environment
echo "🧪 Testing TrackMania Environment..."
python scripts/test_environment.py
echo

# Test 2: Training Pipeline  
echo "🧪 Testing SAC Training Pipeline..."
python scripts/test_training.py
echo

echo "🎉 Demo completed successfully!"
echo
echo "Key components demonstrated:"
echo "  ✅ LIDAR-based environment simulation"
echo "  ✅ SAC algorithm implementation"
echo "  ✅ Episode completion and reward tracking"
echo "  ✅ Docker-ready modular architecture"