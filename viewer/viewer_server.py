#!/usr/bin/env python3
"""
TrackMania Viewer Server
Handles beautiful race replay visualization for demonstrations
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import aiohttp
from dataclasses import dataclass
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Agent:
    id: str
    color: str
    position: Dict[str, float]
    rotation: Dict[str, float]
    speed: float
    trail: List[Dict[str, float]]
    active: bool = True

class RaceViewer:
    """Handles race visualization and replay"""
    
    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.agents: Dict[str, Agent] = {}
        self.current_race_id: Optional[str] = None
        self.current_frame = 0
        self.race_data: Optional[Dict] = None
        self.playing = False
        self.playback_speed = 1.0
        self.colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F"]
        self.color_index = 0
    
    def add_agent(self, agent_id: str) -> Agent:
        """Add agent with unique color"""
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        
        agent = Agent(
            id=agent_id,
            color=color,
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            rotation={"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            speed=0.0,
            trail=[]
        )
        
        self.agents[agent_id] = agent
        logger.info(f"Added agent {agent_id} with color {color}")
        return agent
    
    def update_agent(self, agent_id: str, state: Dict) -> None:
        """Update agent state and trail"""
        if agent_id not in self.agents:
            self.add_agent(agent_id)
        
        agent = self.agents[agent_id]
        agent.position = state["position"]
        agent.rotation = state["rotation"]
        agent.speed = state["speed"]
        
        # Add to trail (limit trail length)
        agent.trail.append(state["position"].copy())
        if len(agent.trail) > 200:  # Keep last 200 positions
            agent.trail.pop(0)
    
    def load_race(self, race_id: str) -> bool:
        """Load race data for replay"""
        race_file = self.data_dir / f"race_{race_id}.json"
        if not race_file.exists():
            return False
        
        with open(race_file, 'r') as f:
            self.race_data = json.load(f)
        
        self.current_race_id = race_id
        self.current_frame = 0
        self.playing = False
        
        # Reset agents
        self.agents.clear()
        self.color_index = 0
        
        logger.info(f"Loaded race {race_id} with {len(self.race_data['frames'])} frames")
        return True
    
    def get_frame_data(self, frame_id: int) -> Dict:
        """Get data for specific frame"""
        if not self.race_data or frame_id >= len(self.race_data["frames"]):
            return {"agents": [], "frame_id": frame_id, "timestamp": ""}
        
        frame = self.race_data["frames"][frame_id]
        agents_data = []
        
        for agent_state in frame["agents"]:
            agent_id = agent_state["agent_id"]
            
            # Update or create agent
            if agent_id not in self.agents:
                self.add_agent(agent_id)
            
            self.update_agent(agent_id, agent_state)
            
            agents_data.append({
                "id": agent_id,
                "color": self.agents[agent_id].color,
                "position": agent_state["position"],
                "rotation": agent_state["rotation"],
                "speed": agent_state["speed"],
                "trail": self.agents[agent_id].trail[-50:],  # Last 50 trail points
                "track_completion": agent_state["track_completion"],
                "lap_time": agent_state["lap_time"]
            })
        
        return {
            "agents": agents_data,
            "frame_id": frame_id,
            "timestamp": frame["timestamp"],
            "total_frames": len(self.race_data["frames"]),
            "playing": self.playing,
            "playback_speed": self.playback_speed
        }
    
    def play(self) -> None:
        """Start playback"""
        self.playing = True
        logger.info("Started playback")
    
    def pause(self) -> None:
        """Pause playback"""
        self.playing = False
        logger.info("Paused playback")
    
    def set_frame(self, frame_id: int) -> None:
        """Jump to specific frame"""
        if self.race_data:
            self.current_frame = max(0, min(frame_id, len(self.race_data["frames"]) - 1))
    
    def set_speed(self, speed: float) -> None:
        """Set playback speed"""
        self.playback_speed = max(0.1, min(speed, 5.0))
        logger.info(f"Set playback speed to {self.playback_speed}x")

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                self.disconnect(connection)

# Global instances
viewer = RaceViewer()
manager = ConnectionManager()

# FastAPI app
app = FastAPI(title="TrackMania Viewer Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send current frame data
            if viewer.race_data and viewer.playing:
                frame_data = viewer.get_frame_data(viewer.current_frame)
                await websocket.send_json(frame_data)
                
                # Advance frame
                viewer.current_frame += 1
                if viewer.current_frame >= len(viewer.race_data["frames"]):
                    viewer.playing = False
                    viewer.current_frame = 0
            
            # Wait based on playback speed
            await asyncio.sleep(0.05 / viewer.playback_speed)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/races")
async def list_races():
    """List available races"""
    races = []
    for race_file in viewer.data_dir.glob("race_*.json"):
        race_id = race_file.stem.replace("race_", "")
        try:
            with open(race_file, 'r') as f:
                race_data = json.load(f)
            races.append({
                "id": race_id,
                "duration": race_data.get("duration", 0),
                "frames": race_data.get("total_frames", 0),
                "recorded_at": race_data.get("recorded_at", "")
            })
        except:
            continue
    
    return {"races": sorted(races, key=lambda x: x["recorded_at"], reverse=True)}

@app.post("/api/load/{race_id}")
async def load_race(race_id: str):
    """Load race for viewing"""
    success = viewer.load_race(race_id)
    if success:
        return {"status": "loaded", "race_id": race_id}
    else:
        raise HTTPException(status_code=404, detail="Race not found")

@app.post("/api/play")
async def play_race():
    """Start race playback"""
    viewer.play()
    await manager.broadcast({"action": "play"})
    return {"status": "playing"}

@app.post("/api/pause")
async def pause_race():
    """Pause race playback"""
    viewer.pause()
    await manager.broadcast({"action": "pause"})
    return {"status": "paused"}

@app.post("/api/frame/{frame_id}")
async def set_frame(frame_id: int):
    """Jump to specific frame"""
    viewer.set_frame(frame_id)
    frame_data = viewer.get_frame_data(viewer.current_frame)
    await manager.broadcast(frame_data)
    return {"status": "frame_set", "frame_id": frame_id}

@app.post("/api/speed/{speed}")
async def set_speed(speed: float):
    """Set playback speed"""
    viewer.set_speed(speed)
    await manager.broadcast({"action": "speed_change", "speed": speed})
    return {"status": "speed_set", "speed": speed}

@app.get("/api/current")
async def get_current_frame():
    """Get current frame data"""
    if viewer.race_data:
        return viewer.get_frame_data(viewer.current_frame)
    else:
        return {"error": "No race loaded"}

@app.get("/api/status")
async def get_status():
    """Get viewer status"""
    return {
        "current_race": viewer.current_race_id,
        "current_frame": viewer.current_frame,
        "playing": viewer.playing,
        "playback_speed": viewer.playback_speed,
        "total_frames": len(viewer.race_data["frames"]) if viewer.race_data else 0,
        "agents": len(viewer.agents),
        "connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def get_viewer():
    """Main viewer interface"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>TrackMania Race Viewer</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; background: #1a1a1a; color: white; }
        .container { display: flex; height: 100vh; }
        .sidebar { width: 300px; background: #2a2a2a; padding: 20px; overflow-y: auto; }
        .main { flex: 1; display: flex; flex-direction: column; }
        .viewer { flex: 1; background: #000; position: relative; }
        .controls { height: 80px; background: #333; display: flex; align-items: center; padding: 0 20px; gap: 10px; }
        .race-item { background: #3a3a3a; padding: 10px; margin: 10px 0; border-radius: 5px; cursor: pointer; }
        .race-item:hover { background: #4a4a4a; }
        .agent { position: absolute; width: 20px; height: 20px; border-radius: 50%; transform: translate(-50%, -50%); }
        .trail { position: absolute; width: 2px; height: 2px; border-radius: 50%; opacity: 0.6; }
        button { padding: 8px 15px; background: #007acc; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #005999; }
        button:disabled { background: #666; cursor: not-allowed; }
        input[type="range"] { flex: 1; margin: 0 10px; }
        .info { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; }
        .speed-display { min-width: 50px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Race Replays</h2>
            <div id="raceList">Loading...</div>
        </div>
        <div class="main">
            <div class="viewer" id="viewer">
                <div class="info" id="info">
                    <div>Frame: <span id="frameInfo">0/0</span></div>
                    <div>Time: <span id="timeInfo">0.00s</span></div>
                    <div>Agents: <span id="agentInfo">0</span></div>
                </div>
            </div>
            <div class="controls">
                <button id="playBtn" onclick="togglePlay()">Play</button>
                <button onclick="setSpeed(0.5)">0.5x</button>
                <button onclick="setSpeed(1.0)">1x</button>
                <button onclick="setSpeed(2.0)">2x</button>
                <button onclick="setSpeed(4.0)">4x</button>
                <div class="speed-display" id="speedDisplay">1.0x</div>
                <input type="range" id="frameSlider" min="0" max="100" value="0" onchange="setFrame(this.value)">
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let currentRace = null;
        let playing = false;
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateViewer(data);
            };
            
            ws.onclose = function() {
                setTimeout(connectWebSocket, 1000);
            };
        }
        
        function updateViewer(data) {
            if (data.agents) {
                const viewer = document.getElementById('viewer');
                const viewerRect = viewer.getBoundingClientRect();
                
                // Clear previous agents
                viewer.querySelectorAll('.agent, .trail').forEach(el => el.remove());
                
                // Draw agents and trails
                data.agents.forEach(agent => {
                    // Draw trail
                    agent.trail.forEach((pos, index) => {
                        const trail = document.createElement('div');
                        trail.className = 'trail';
                        trail.style.backgroundColor = agent.color;
                        trail.style.left = (pos.x * 2 + viewerRect.width/2) + 'px';
                        trail.style.top = (pos.y * 2 + viewerRect.height/2) + 'px';
                        trail.style.opacity = (index / agent.trail.length) * 0.6;
                        viewer.appendChild(trail);
                    });
                    
                    // Draw agent
                    const agentEl = document.createElement('div');
                    agentEl.className = 'agent';
                    agentEl.style.backgroundColor = agent.color;
                    agentEl.style.left = (agent.position.x * 2 + viewerRect.width/2) + 'px';
                    agentEl.style.top = (agent.position.y * 2 + viewerRect.height/2) + 'px';
                    agentEl.title = `${agent.id}: ${agent.speed.toFixed(1)} km/h`;
                    viewer.appendChild(agentEl);
                });
                
                // Update info
                document.getElementById('frameInfo').textContent = `${data.frame_id}/${data.total_frames}`;
                document.getElementById('timeInfo').textContent = (data.frame_id * 0.05).toFixed(2) + 's';
                document.getElementById('agentInfo').textContent = data.agents.length;
                
                // Update slider
                const slider = document.getElementById('frameSlider');
                slider.max = data.total_frames - 1;
                slider.value = data.frame_id;
                
                playing = data.playing;
                document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
            }
        }
        
        async function loadRaces() {
            try {
                const response = await fetch('/api/races');
                const data = await response.json();
                
                const raceList = document.getElementById('raceList');
                raceList.innerHTML = '';
                
                data.races.forEach(race => {
                    const item = document.createElement('div');
                    item.className = 'race-item';
                    item.innerHTML = `
                        <div><strong>Race ${race.id.substring(0, 8)}</strong></div>
                        <div>${race.frames} frames, ${race.duration.toFixed(1)}s</div>
                        <div><small>${new Date(race.recorded_at).toLocaleString()}</small></div>
                    `;
                    item.onclick = () => loadRace(race.id);
                    raceList.appendChild(item);
                });
            } catch (error) {
                document.getElementById('raceList').textContent = 'Error loading races';
            }
        }
        
        async function loadRace(raceId) {
            try {
                await fetch(`/api/load/${raceId}`, { method: 'POST' });
                currentRace = raceId;
                document.title = `TrackMania Race Viewer - Race ${raceId.substring(0, 8)}`;
            } catch (error) {
                alert('Error loading race');
            }
        }
        
        async function togglePlay() {
            if (!currentRace) {
                alert('Please load a race first');
                return;
            }
            
            const endpoint = playing ? '/api/pause' : '/api/play';
            await fetch(endpoint, { method: 'POST' });
        }
        
        async function setSpeed(speed) {
            await fetch(`/api/speed/${speed}`, { method: 'POST' });
            document.getElementById('speedDisplay').textContent = speed + 'x';
        }
        
        async function setFrame(frameId) {
            await fetch(`/api/frame/${frameId}`, { method: 'POST' });
        }
        
        // Initialize
        connectWebSocket();
        loadRaces();
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    logger.info("Starting TrackMania Viewer Server...")
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")