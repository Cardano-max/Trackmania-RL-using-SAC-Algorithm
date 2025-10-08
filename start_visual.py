#!/usr/bin/env python3
"""
Simple script to start visual interface
"""

import subprocess
import time
import os
from pathlib import Path

# Create data directory
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)
print(f"✅ Created data directory: {data_dir}")

# Set up environment
env = os.environ.copy()
env['PYTHONPATH'] = '.'

def start_server(script, port, name):
    """Start a server"""
    print(f"🚀 Starting {name} on port {port}...")
    try:
        process = subprocess.Popen([
            "python3", script
        ], env=env, cwd=".")
        return process
    except Exception as e:
        print(f"❌ Error starting {name}: {e}")
        return None

def main():
    print("🎮 TrackMania RL Visual Interface")
    print("=" * 40)
    
    # Start environment server
    env_process = start_server("environment/environment_server.py", 8080, "Environment Server")
    time.sleep(3)
    
    # Start model server
    model_process = start_server("model/model_server.py", 8081, "Model Server")
    time.sleep(3)
    
    # Start viewer server
    viewer_process = start_server("viewer/viewer_server.py", 3000, "Viewer Server")
    time.sleep(2)
    
    print("\n📋 Servers Status:")
    print("   🎮 Environment API: http://localhost:8080")
    print("   🤖 Model API: http://localhost:8081")
    print("   🎬 Viewer Interface: http://localhost:3000")
    print("\n🎯 Open your browser to: http://localhost:3000")
    print("💡 Press Ctrl+C to stop")
    
    try:
        # Wait for user to stop
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        if env_process:
            env_process.terminate()
        if model_process:
            model_process.terminate()
        if viewer_process:
            viewer_process.terminate()

if __name__ == "__main__":
    main()