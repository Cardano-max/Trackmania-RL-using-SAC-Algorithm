#!/bin/bash

echo "🔍 TrackMania RL Dashboard Status Check"
echo "========================================="

echo "📊 Process Status:"
ps aux | grep learning_dashboard | grep -v grep

echo ""
echo "🌐 Server Status:"
curl -s http://localhost:4000/api/status | head -c 200
echo ""

echo ""
echo "📝 To see live logs, run:"
echo "   ps aux | grep learning_dashboard"
echo "   kill -USR1 <PID>  # to get detailed logs"

echo ""
echo "🎯 Dashboard URL: http://localhost:4000"
echo "📋 The dashboard is running and ready for training!"