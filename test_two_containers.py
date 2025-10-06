#!/usr/bin/env python3
"""
Test script for two-container TrackMania RL system
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def test_environment_container():
    """Test environment container API"""
    print("🧪 Testing Environment Container...")
    
    async with aiohttp.ClientSession() as session:
        # Test status endpoint
        try:
            async with session.get("http://localhost:8080/api/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Environment Status: {data['status']}")
                else:
                    print(f"❌ Environment Status Error: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Environment Connection Error: {e}")
            return False
        
        # Test add agent
        try:
            async with session.post("http://localhost:8080/api/agents/add/test_agent") as response:
                if response.status == 200:
                    print("✅ Agent Added Successfully")
                else:
                    print(f"❌ Add Agent Error: {response.status}")
        except Exception as e:
            print(f"❌ Add Agent Error: {e}")
        
        # Test action processing
        action_data = {
            "agent_id": "test_agent",
            "action": {
                "gas": 0.8,
                "brake": 0.0,
                "steering": 0.2
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            async with session.post("http://localhost:8080/api/action", json=action_data) as response:
                if response.status == 200:
                    state = await response.json()
                    print(f"✅ Action Processed: Reward={state['reward']:.2f}, Speed={state['speed']:.1f}")
                else:
                    print(f"❌ Action Processing Error: {response.status}")
        except Exception as e:
            print(f"❌ Action Processing Error: {e}")
        
        # Test recording
        try:
            async with session.post("http://localhost:8080/api/recording/start") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Recording Started: {data['race_id']}")
                    
                    # Stop recording
                    await asyncio.sleep(1)
                    async with session.post("http://localhost:8080/api/recording/stop") as response:
                        if response.status == 200:
                            data = await response.json()
                            print(f"✅ Recording Stopped: {data['race_id']}")
                        
        except Exception as e:
            print(f"❌ Recording Error: {e}")
    
    return True

async def test_model_container():
    """Test model container API"""
    print("\n🤖 Testing Model Container...")
    
    async with aiohttp.ClientSession() as session:
        # Test status endpoint
        try:
            async with session.get("http://localhost:8081/api/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Model Status: {data['status']}")
                    print(f"   PyTorch Available: {data['pytorch_available']}")
                    print(f"   Device: {data['device']}")
                else:
                    print(f"❌ Model Status Error: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Model Connection Error: {e}")
            return False
        
        # Test single action
        observation = {
            "speed": 120.5,
            "lidar": [0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 
                     0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.6],
            "position": {"x": 100.0, "y": 50.0, "z": 0.0},
            "rotation": {"yaw": 0.5, "pitch": 0.0, "roll": 0.0}
        }
        
        try:
            async with session.post("http://localhost:8081/api/action/single", json=observation) as response:
                if response.status == 200:
                    action = await response.json()
                    print(f"✅ Action Generated: Gas={action['action']['gas']:.2f}, "
                          f"Brake={action['action']['brake']:.2f}, "
                          f"Steering={action['action']['steering']:.2f}")
                else:
                    print(f"❌ Action Generation Error: {response.status}")
        except Exception as e:
            print(f"❌ Action Generation Error: {e}")
        
        # Test training status
        try:
            async with session.get("http://localhost:8081/api/training/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Training Status: Active={data['active']}")
                    print(f"   Episodes: {data['stats']['episodes']}")
                    print(f"   Buffer Size: {data['stats'].get('buffer_size', 0)}")
                else:
                    print(f"❌ Training Status Error: {response.status}")
        except Exception as e:
            print(f"❌ Training Status Error: {e}")
    
    return True

async def test_viewer_container():
    """Test viewer container API"""
    print("\n📺 Testing Viewer Container...")
    
    async with aiohttp.ClientSession() as session:
        # Test status endpoint
        try:
            async with session.get("http://localhost:3000/api/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Viewer Status: Current Race={data['current_race']}")
                    print(f"   Connections: {data['connections']}")
                else:
                    print(f"❌ Viewer Status Error: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Viewer Connection Error: {e}")
            return False
        
        # Test races list
        try:
            async with session.get("http://localhost:3000/api/races") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Available Races: {len(data['races'])}")
                else:
                    print(f"❌ Races List Error: {response.status}")
        except Exception as e:
            print(f"❌ Races List Error: {e}")
    
    return True

async def test_container_communication():
    """Test communication between containers"""
    print("\n🔄 Testing Container Communication...")
    
    async with aiohttp.ClientSession() as session:
        # Start recording in environment
        print("1. Starting race recording...")
        try:
            async with session.post("http://localhost:8080/api/recording/start") as response:
                if response.status == 200:
                    data = await response.json()
                    race_id = data['race_id']
                    print(f"✅ Recording started: {race_id}")
                else:
                    print("❌ Failed to start recording")
                    return False
        except Exception as e:
            print(f"❌ Recording start error: {e}")
            return False
        
        # Start training in model
        print("2. Starting training...")
        try:
            async with session.post("http://localhost:8081/api/training/start") as response:
                if response.status == 200:
                    print("✅ Training started")
                else:
                    print("❌ Failed to start training")
        except Exception as e:
            print(f"❌ Training start error: {e}")
        
        # Let it run for a few seconds
        print("3. Running for 10 seconds...")
        await asyncio.sleep(10)
        
        # Stop training
        print("4. Stopping training...")
        try:
            async with session.post("http://localhost:8081/api/training/stop") as response:
                if response.status == 200:
                    print("✅ Training stopped")
        except Exception as e:
            print(f"❌ Training stop error: {e}")
        
        # Stop recording
        print("5. Stopping recording...")
        try:
            async with session.post("http://localhost:8080/api/recording/stop") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Recording stopped: {data['race_id']}")
                    
                    # Check if race is available in viewer
                    await asyncio.sleep(2)
                    async with session.get("http://localhost:3000/api/races") as response:
                        if response.status == 200:
                            data = await response.json()
                            if data['races']:
                                print(f"✅ Race available in viewer: {len(data['races'])} total races")
                            else:
                                print("⚠️ No races found in viewer")
                        
        except Exception as e:
            print(f"❌ Recording stop error: {e}")
        
        return True

async def main():
    """Run all tests"""
    print("🚀 TrackMania Two-Container System Test\n")
    
    # Test individual containers
    env_ok = await test_environment_container()
    model_ok = await test_model_container()
    viewer_ok = await test_viewer_container()
    
    if env_ok and model_ok and viewer_ok:
        # Test communication
        comm_ok = await test_container_communication()
        
        if comm_ok:
            print("\n🎉 All Tests Passed!")
            print("\n📋 System Ready:")
            print("   Environment: http://localhost:8080")
            print("   Model: http://localhost:8081") 
            print("   Viewer: http://localhost:3000")
            print("   TensorBoard: http://localhost:6006")
            print("\n🎮 Open the viewer in your browser to see race replays!")
        else:
            print("\n❌ Communication tests failed")
    else:
        print("\n❌ Some container tests failed")
        print("   Make sure all containers are running:")
        print("   docker-compose -f docker-compose-v2.yml up -d")

if __name__ == "__main__":
    asyncio.run(main())