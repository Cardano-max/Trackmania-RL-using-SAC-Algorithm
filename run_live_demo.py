#!/usr/bin/env python3
"""
Create live racing simulation with real data
"""

import asyncio
import aiohttp
import json
import time
import numpy as np

async def create_live_race():
    """Create a live racing simulation"""
    
    # Start recording
    async with aiohttp.ClientSession() as session:
        print("🎬 Starting live race recording...")
        
        # Start recording
        async with session.post("http://localhost:8080/api/recording/start") as response:
            if response.status == 200:
                data = await response.json()
                race_id = data["race_id"]
                print(f"✅ Started recording race: {race_id}")
            else:
                print("❌ Failed to start recording")
                return
        
        # Register 3 agents
        agents = [
            {"agent_id": "racer_1", "name": "Blue Lightning", "color": "#3498db"},
            {"agent_id": "racer_2", "name": "Red Storm", "color": "#e74c3c"}, 
            {"agent_id": "racer_3", "name": "Green Thunder", "color": "#2ecc71"}
        ]
        
        for agent in agents:
            async with session.post("http://localhost:8080/api/agents/register", json=agent) as response:
                if response.status == 200:
                    print(f"✅ Registered {agent['name']}")
                else:
                    print(f"❌ Failed to register {agent['name']}")
        
        print("🏁 Starting race simulation...")
        
        # Simulate 30 seconds of racing (600 frames at 20Hz)
        for frame in range(600):
            time_sec = frame * 0.05  # 20Hz = 0.05s per frame
            
            # Send actions for each agent
            for i, agent in enumerate(agents):
                # Create realistic racing behavior
                base_speed = 25 + i * 5  # Different base speeds
                
                # Add some realistic variation
                speed_variation = 10 * np.sin(time_sec * 0.3 + i)
                steering_input = 0.3 * np.sin(time_sec * 0.8 + i * 1.5)
                
                # Gas/brake logic
                if frame % 60 == 0:  # Occasional braking
                    gas = max(0.3, 0.8 - np.random.random() * 0.4)
                    brake = np.random.random() * 0.3
                else:
                    gas = min(1.0, 0.7 + np.random.random() * 0.3)
                    brake = 0.0
                
                action = {
                    "agent_id": agent["agent_id"],
                    "action": {
                        "gas": gas,
                        "brake": brake,
                        "steering": np.clip(steering_input, -1, 1)
                    }
                }
                
                # Send action to environment
                try:
                    async with session.post("http://localhost:8080/api/action", json=action) as response:
                        if response.status == 200:
                            state_data = await response.json()
                            if frame % 100 == 0:  # Log every 5 seconds
                                print(f"🏎️  {agent['name']}: Speed={state_data.get('speed', 0):.1f}, Reward={state_data.get('reward', 0):.2f}")
                except Exception as e:
                    print(f"⚠️  Action error for {agent['name']}: {e}")
            
            # Small delay to maintain 20Hz
            await asyncio.sleep(0.05)
            
            if frame % 100 == 0:
                print(f"📊 Frame {frame}/600 - {time_sec:.1f}s elapsed")
        
        print("🏁 Race finished! Stopping recording...")
        
        # Stop recording
        async with session.post("http://localhost:8080/api/recording/stop") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Recording stopped. Race ID: {data.get('race_id')}")
                print(f"🎯 Open browser to: http://localhost:3000")
                print(f"🎬 Your race is ready for replay!")
            else:
                print("❌ Failed to stop recording")

if __name__ == "__main__":
    print("🚀 TrackMania Live Racing Demo")
    print("=" * 40)
    print("📋 Servers should be running on:")
    print("   🎮 Environment: http://localhost:8080")
    print("   🤖 Model: http://localhost:8081") 
    print("   🎬 Viewer: http://localhost:3000")
    print("=" * 40)
    
    asyncio.run(create_live_race())