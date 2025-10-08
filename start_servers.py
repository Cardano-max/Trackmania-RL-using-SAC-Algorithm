#!/usr/bin/env python3
"""
Start all servers locally to show visual interface
"""

import subprocess
import time
import threading
import sys
import os
from pathlib import Path

# Ensure we have the data directory
data_dir = Path("/tmp/trackmania_data")
data_dir.mkdir(exist_ok=True)

def run_environment_server():
    """Run environment server"""
    print("🎮 Starting Environment Server on port 8080...")
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    try:
        subprocess.run([
            sys.executable, "environment/environment_server.py"
        ], env=env, cwd=".")
    except Exception as e:
        print(f"Environment server error: {e}")

def run_model_server():
    """Run model server"""
    print("🤖 Starting Model Server on port 8081...")
    time.sleep(2)  # Wait for environment server
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    try:
        subprocess.run([
            sys.executable, "model/model_server.py"
        ], env=env, cwd=".")
    except Exception as e:
        print(f"Model server error: {e}")

def run_viewer_server():
    """Run viewer server"""
    print("🎬 Starting Viewer Server on port 3000...")
    time.sleep(4)  # Wait for other servers
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    try:
        subprocess.run([
            sys.executable, "viewer/viewer_server.py"
        ], env=env, cwd=".")
    except Exception as e:
        print(f"Viewer server error: {e}")

def main():
    print("🚀 Starting TrackMania RL Visual Interface...")
    print("=" * 50)
    
    # Start servers in separate threads
    threads = []
    
    # Environment server
    env_thread = threading.Thread(target=run_environment_server, daemon=True)
    env_thread.start()
    threads.append(env_thread)
    
    # Model server  
    model_thread = threading.Thread(target=run_model_server, daemon=True)
    model_thread.start()
    threads.append(model_thread)
    
    # Viewer server
    viewer_thread = threading.Thread(target=run_viewer_server, daemon=True)
    viewer_thread.start()
    threads.append(viewer_thread)
    
    print("\n📋 Servers starting...")
    print("   🎮 Environment API: http://localhost:8080")
    print("   🤖 Model API: http://localhost:8081")
    print("   🎬 Viewer Interface: http://localhost:3000")
    print("\n⏳ Wait 10 seconds then open: http://localhost:3000")
    print("💡 Press Ctrl+C to stop all servers")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        return

if __name__ == "__main__":
    main()