#!/usr/bin/env python3
"""
TrackMania Environment Server
Handles simulation, graphics, and communication with model container
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import threading
import queue
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentAction:
    agent_id: str
    gas: float
    brake: float
    steering: float
    timestamp: str

@dataclass
class AgentState:
    agent_id: str
    speed: float
    position: Dict[str, float]  # x, y, z
    rotation: Dict[str, float]  # yaw, pitch, roll
    lidar: List[float]
    reward: float
    done: bool
    track_completion: float
    lap_time: float
    timestamp: str

@dataclass
class RaceFrame:
    timestamp: str
    agents: List[AgentState]
    frame_id: int

class TrackManiaEnvironment:
    """Core environment simulation"""
    
    def __init__(self):
        self.agents: Dict[str, Dict] = {}
        self.track_length = 1000.0
        self.track_width = 20.0
        self.reset_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.reset_rotation = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        
    def add_agent(self, agent_id: str) -> None:
        """Add new agent to environment"""
        self.agents[agent_id] = {
            "position": self.reset_position.copy(),
            "rotation": self.reset_rotation.copy(),
            "speed": 0.0,
            "last_action": {"gas": 0.0, "brake": 0.0, "steering": 0.0},
            "episode_time": 0.0,
            "track_progress": 0.0,
            "total_reward": 0.0,
            "episode_steps": 0
        }
        logger.info(f"Added agent: {agent_id}")
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from environment"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Removed agent: {agent_id}")
            return True
        return False
    
    def step(self, action: AgentAction) -> AgentState:
        """Process agent action and return new state"""
        if action.agent_id not in self.agents:
            self.add_agent(action.agent_id)
        
        agent = self.agents[action.agent_id]
        
        # Update physics
        dt = 0.05  # 20Hz simulation
        
        # Speed dynamics
        acceleration = action.gas * 300.0 - action.brake * 500.0  # max speed ~300 km/h
        agent["speed"] = max(0, min(300, agent["speed"] + acceleration * dt))
        
        # Position update
        speed_ms = agent["speed"] / 3.6  # km/h to m/s
        yaw = agent["rotation"]["yaw"]
        
        dx = speed_ms * np.cos(yaw) * dt
        dy = speed_ms * np.sin(yaw) * dt
        
        agent["position"]["x"] += dx
        agent["position"]["y"] += dy
        
        # Rotation update (steering)
        steering_sensitivity = 2.0
        if agent["speed"] > 5:  # Only steer when moving
            agent["rotation"]["yaw"] += action.steering * steering_sensitivity * dt
        
        # Track progress calculation
        distance = np.sqrt(agent["position"]["x"]**2 + agent["position"]["y"]**2)
        agent["track_progress"] = min(1.0, distance / self.track_length)
        
        # LIDAR simulation
        lidar = self._simulate_lidar(agent)
        
        # Reward calculation
        reward = self._calculate_reward(agent, action)
        agent["total_reward"] += reward
        
        # Episode management
        agent["episode_time"] += dt
        agent["episode_steps"] += 1
        done = self._check_episode_done(agent)
        
        # Store last action for physics
        agent["last_action"] = {
            "gas": action.gas,
            "brake": action.brake, 
            "steering": action.steering
        }
        
        return AgentState(
            agent_id=action.agent_id,
            speed=agent["speed"],
            position=agent["position"].copy(),
            rotation=agent["rotation"].copy(),
            lidar=lidar,
            reward=reward,
            done=done,
            track_completion=agent["track_progress"],
            lap_time=agent["episode_time"],
            timestamp=datetime.now().isoformat()
        )
    
    def _simulate_lidar(self, agent: Dict) -> List[float]:
        """Simulate 19-beam LIDAR sensor"""
        lidar_beams = []
        agent_pos = agent["position"]
        agent_yaw = agent["rotation"]["yaw"]
        
        for i in range(19):
            beam_angle = (i - 9) * 0.1 + agent_yaw  # ±45 degrees
            
            # Simple track boundary detection
            beam_x = agent_pos["x"] + 50 * np.cos(beam_angle)
            beam_y = agent_pos["y"] + 50 * np.sin(beam_angle)
            
            # Distance to track boundary (simplified)
            track_center_distance = abs(beam_y)
            if track_center_distance > self.track_width / 2:
                distance = 0.1 + np.random.uniform(0, 0.2)  # Close to boundary
            else:
                distance = 0.8 + np.random.uniform(0, 0.2)  # On track
            
            lidar_beams.append(distance)
        
        return lidar_beams
    
    def _calculate_reward(self, agent: Dict, action: AgentAction) -> float:
        """Calculate reward for current step"""
        reward = 0.0
        
        # Progress reward
        progress_reward = agent["track_progress"] * 100.0
        
        # Speed reward (encourage optimal speed)
        speed_reward = min(agent["speed"] / 150.0, 1.0) * 10.0
        
        # Steering penalty (encourage smooth driving)
        steering_penalty = -abs(action.steering) * 5.0
        
        # Track boundary penalty
        if agent["position"]["y"] > self.track_width / 2:
            boundary_penalty = -20.0
        else:
            boundary_penalty = 5.0
        
        reward = progress_reward + speed_reward + steering_penalty + boundary_penalty
        return reward
    
    def _check_episode_done(self, agent: Dict) -> bool:
        """Check if episode should terminate"""
        # Complete track
        if agent["track_progress"] >= 1.0:
            return True
        
        # Time limit
        if agent["episode_time"] > 60.0:  # 60 seconds max
            return True
        
        # Out of bounds
        if abs(agent["position"]["y"]) > self.track_width:
            return True
        
        return False
    
    def reset_agent(self, agent_id: str) -> AgentState:
        """Reset specific agent"""
        if agent_id not in self.agents:
            self.add_agent(agent_id)
        
        agent = self.agents[agent_id]
        agent.update({
            "position": self.reset_position.copy(),
            "rotation": self.reset_rotation.copy(),
            "speed": 0.0,
            "episode_time": 0.0,
            "track_progress": 0.0,
            "total_reward": 0.0,
            "episode_steps": 0
        })
        
        return AgentState(
            agent_id=agent_id,
            speed=0.0,
            position=agent["position"].copy(),
            rotation=agent["rotation"].copy(),
            lidar=self._simulate_lidar(agent),
            reward=0.0,
            done=False,
            track_completion=0.0,
            lap_time=0.0,
            timestamp=datetime.now().isoformat()
        )

class RaceRecorder:
    """Records and manages race replays"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.current_race_id: Optional[str] = None
        self.race_frames: List[RaceFrame] = []
        self.frame_counter = 0
        
    def start_recording(self, race_id: Optional[str] = None) -> str:
        """Start recording new race"""
        self.current_race_id = race_id or str(uuid.uuid4())
        self.race_frames = []
        self.frame_counter = 0
        logger.info(f"Started recording race: {self.current_race_id}")
        return self.current_race_id
    
    def record_frame(self, agents_states: List[AgentState]) -> None:
        """Record current frame"""
        if self.current_race_id is None:
            return
        
        frame = RaceFrame(
            timestamp=datetime.now().isoformat(),
            agents=agents_states,
            frame_id=self.frame_counter
        )
        self.race_frames.append(frame)
        self.frame_counter += 1
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and save race"""
        if self.current_race_id is None:
            return None
        
        race_file = self.data_dir / f"race_{self.current_race_id}.json"
        race_data = {
            "race_id": self.current_race_id,
            "frames": [asdict(frame) for frame in self.race_frames],
            "total_frames": len(self.race_frames),
            "duration": len(self.race_frames) * 0.05,  # 20Hz
            "recorded_at": datetime.now().isoformat()
        }
        
        with open(race_file, 'w') as f:
            json.dump(race_data, f, indent=2)
        
        logger.info(f"Saved race {self.current_race_id} with {len(self.race_frames)} frames")
        race_id = self.current_race_id
        self.current_race_id = None
        return race_id
    
    def get_race(self, race_id: str) -> Optional[Dict]:
        """Load race data"""
        race_file = self.data_dir / f"race_{race_id}.json"
        if race_file.exists():
            with open(race_file, 'r') as f:
                return json.load(f)
        return None
    
    def list_races(self) -> List[str]:
        """List available races"""
        races = []
        for race_file in self.data_dir.glob("race_*.json"):
            race_id = race_file.stem.replace("race_", "")
            races.append(race_id)
        return sorted(races)

# Global instances
env = TrackManiaEnvironment()
recorder = RaceRecorder()

# FastAPI app
app = FastAPI(title="TrackMania Environment Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/action")
async def process_action(action_data: dict) -> dict:
    """Process agent action and return environment state"""
    try:
        action = AgentAction(
            agent_id=action_data["agent_id"],
            gas=action_data["action"]["gas"],
            brake=action_data["action"]["brake"],
            steering=action_data["action"]["steering"],
            timestamp=action_data.get("timestamp", datetime.now().isoformat())
        )
        
        state = env.step(action)
        
        # Record frame if recording
        if recorder.current_race_id:
            recorder.record_frame([state])
        
        return asdict(state)
    
    except Exception as e:
        logger.error(f"Error processing action: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/reset/{agent_id}")
async def reset_agent(agent_id: str) -> dict:
    """Reset specific agent"""
    try:
        state = env.reset_agent(agent_id)
        return asdict(state)
    except Exception as e:
        logger.error(f"Error resetting agent {agent_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/agents/add/{agent_id}")
async def add_agent(agent_id: str) -> dict:
    """Add new agent"""
    env.add_agent(agent_id)
    return {"status": "success", "agent_id": agent_id}

@app.delete("/api/agents/{agent_id}")
async def remove_agent(agent_id: str) -> dict:
    """Remove agent"""
    success = env.remove_agent(agent_id)
    if success:
        return {"status": "success", "agent_id": agent_id}
    else:
        raise HTTPException(status_code=404, detail="Agent not found")

@app.get("/api/agents")
async def list_agents() -> dict:
    """List all active agents"""
    return {"agents": list(env.agents.keys())}

@app.post("/api/recording/start")
async def start_recording(race_id: str = None) -> dict:
    """Start race recording"""
    race_id = recorder.start_recording(race_id)
    return {"status": "recording", "race_id": race_id}

@app.post("/api/recording/stop")
async def stop_recording() -> dict:
    """Stop race recording"""
    race_id = recorder.stop_recording()
    if race_id:
        return {"status": "stopped", "race_id": race_id}
    else:
        return {"status": "no_recording"}

@app.get("/api/races")
async def list_races() -> dict:
    """List available races"""
    races = recorder.list_races()
    return {"races": races}

@app.get("/api/race/{race_id}")
async def get_race(race_id: str) -> dict:
    """Get race data"""
    race_data = recorder.get_race(race_id)
    if race_data:
        return race_data
    else:
        raise HTTPException(status_code=404, detail="Race not found")

@app.get("/api/status")
async def get_status() -> dict:
    """Get environment status"""
    return {
        "status": "running",
        "agents": len(env.agents),
        "recording": recorder.current_race_id is not None,
        "current_race": recorder.current_race_id,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return HTMLResponse("""
    <html>
        <head><title>TrackMania Environment Server</title></head>
        <body>
            <h1>TrackMania Environment Server</h1>
            <p>Environment container is running!</p>
            <h2>API Endpoints:</h2>
            <ul>
                <li><b>POST /api/action</b> - Process agent action</li>
                <li><b>POST /api/reset/{agent_id}</b> - Reset agent</li>
                <li><b>POST /api/agents/add/{agent_id}</b> - Add agent</li>
                <li><b>GET /api/agents</b> - List agents</li>
                <li><b>POST /api/recording/start</b> - Start recording</li>
                <li><b>POST /api/recording/stop</b> - Stop recording</li>
                <li><b>GET /api/races</b> - List races</li>
                <li><b>GET /api/race/{race_id}</b> - Get race data</li>
                <li><b>GET /api/status</b> - Environment status</li>
            </ul>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """)

if __name__ == "__main__":
    logger.info("Starting TrackMania Environment Server...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")