#!/usr/bin/env python3
"""
Debug script to find exactly what's wrong with the demo
"""

import sys
import os
import subprocess
import requests
import time
from pathlib import Path

def check_environment():
    """Check if we're in the right environment"""
    print("🔍 Environment Check")
    print("=" * 30)
    print(f"Python path: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Virtual env: {os.environ.get('VIRTUAL_ENV', 'None')}")
    
    # Check if required files exist
    required_files = [
        "tmrl_enhanced/learning_dashboard.py",
        "tmrl_enhanced/rl_learning_engine.py",
        "simple_rl_demo.py"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
    
    print()

def check_imports():
    """Check if all required packages are installed"""
    print("📦 Package Check")
    print("=" * 30)
    
    required_packages = [
        "fastapi", "uvicorn", "numpy", "aiohttp", "websockets"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
    
    print()

def test_import_dashboard():
    """Test if we can import the dashboard"""
    print("🧪 Dashboard Import Test")
    print("=" * 30)
    
    try:
        sys.path.append('tmrl_enhanced')
        from learning_dashboard import dashboard
        print("✅ Dashboard import successful")
        
        # Test basic functionality
        data = dashboard.get_dashboard_data()
        print(f"✅ Dashboard data generation works")
        print(f"   Training active: {data['training_active']}")
        print(f"   Current episode: {data['current_episode']}")
        print(f"   Agents: {len(data['agents'])}")
        
    except Exception as e:
        print(f"❌ Dashboard import failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def check_server_status():
    """Check if server is running and responding"""
    print("🌐 Server Status Check")
    print("=" * 30)
    
    # Check if port 4000 is in use
    try:
        result = subprocess.run(['lsof', '-i', ':4000'], 
                              capture_output=True, text=True)
        if result.stdout:
            print("✅ Port 4000 is in use")
            print(f"   Process: {result.stdout.strip()}")
        else:
            print("❌ Port 4000 is not in use")
    except Exception as e:
        print(f"⚠️  Could not check port: {e}")
    
    # Try to connect to server
    try:
        response = requests.get("http://localhost:4000/api/status", timeout=5)
        print(f"✅ Server responding: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Server not responding")
    except Exception as e:
        print(f"❌ Server error: {e}")
    
    print()

def test_training_start():
    """Test starting training via API"""
    print("🚀 Training Start Test")
    print("=" * 30)
    
    try:
        response = requests.post("http://localhost:4000/api/training/start", timeout=10)
        print(f"✅ Training start response: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Wait a moment and check status
        time.sleep(2)
        status_response = requests.get("http://localhost:4000/api/status", timeout=5)
        status_data = status_response.json()
        print(f"   Training active: {status_data.get('training_active', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Training start failed: {e}")
    
    print()

def main():
    print("🔍 TrackMania RL Demo Debugging")
    print("=" * 50)
    print()
    
    check_environment()
    check_imports()
    test_import_dashboard()
    check_server_status()
    test_training_start()
    
    print("🎯 Debug Summary")
    print("=" * 30)
    print("If all checks passed ✅, the demo should work.")
    print("If any checks failed ❌, those need to be fixed first.")
    print()
    print("💡 Next steps:")
    print("1. Fix any failed checks")
    print("2. Restart demo: ./start_client_demo.sh")
    print("3. Open: http://localhost:4000")
    print("4. Click: Start Training")

if __name__ == "__main__":
    main()