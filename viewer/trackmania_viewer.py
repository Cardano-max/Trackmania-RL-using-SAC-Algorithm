#!/usr/bin/env python3
"""
TrackMania Racing Viewer with Proper Track and Cars
"""

import json
import time
import logging
import math
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
class Car:
    agent_id: str
    name: str
    color: str
    position: List[float]
    rotation: float
    speed: float
    lap: int
    lap_time: float
    track_position: float  # 0-1 around track
    active: bool = True

class TrackManiaViewer:
    """Enhanced viewer with proper track and racing visuals"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.cars: Dict[str, Car] = {}
        self.current_race_id: Optional[str] = None
        self.current_frame = 0
        self.race_data: Optional[Dict] = None
        self.playing = False
        self.playback_speed = 1.0
        
        # Track layout - Oval track
        self.track_center_x = 400
        self.track_center_y = 300
        self.track_width = 300
        self.track_height = 200
        
    def load_race(self, race_id: str) -> bool:
        """Load race data"""
        race_file = self.data_dir / f"race_{race_id}.json"
        if not race_file.exists():
            logger.error(f"Race file not found: {race_file}")
            return False
        
        try:
            with open(race_file, 'r') as f:
                self.race_data = json.load(f)
            
            self.current_race_id = race_id
            self.current_frame = 0
            self.playing = False
            
            # Reset cars
            self.cars.clear()
            
            logger.info(f"Loaded race {race_id} with {len(self.race_data['frames'])} frames")
            return True
        except Exception as e:
            logger.error(f"Error loading race: {e}")
            return False
    
    def convert_to_track_position(self, x: float, y: float) -> Dict:
        """Convert simulation position to track coordinates"""
        # Convert the random movements to realistic track following
        frame_progress = self.current_frame / max(1, len(self.race_data.get('frames', [1])))
        
        # Create oval track path
        angle = frame_progress * 4 * math.pi  # 2 full laps
        
        # Add some variation based on x,y
        angle_offset = (x + y) * 0.01
        radius_offset = (x - y) * 0.1
        
        track_x = self.track_center_x + (self.track_width + radius_offset) * math.cos(angle + angle_offset)
        track_y = self.track_center_y + (self.track_height + radius_offset * 0.5) * math.sin(angle + angle_offset)
        
        # Calculate rotation (tangent to track)
        rotation = angle + angle_offset + math.pi/2
        
        return {
            "x": track_x,
            "y": track_y,
            "rotation": rotation,
            "track_progress": (frame_progress * 2) % 1.0  # 0-1 around track
        }
    
    def get_frame_data(self, frame_id: int) -> Dict:
        """Get racing data for specific frame"""
        if not self.race_data or frame_id >= len(self.race_data["frames"]):
            return {"cars": [], "frame_id": frame_id, "timestamp": ""}
        
        frame = self.race_data["frames"][frame_id]
        cars_data = []
        
        for agent_state in frame["agents"]:
            agent_id = agent_state["agent_id"]
            
            # Convert to track position
            track_pos = self.convert_to_track_position(
                agent_state["position"]["x"], 
                agent_state["position"]["y"]
            )
            
            # Update or create car
            if agent_id not in self.cars:
                colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
                color = colors[len(self.cars) % len(colors)]
                
                self.cars[agent_id] = Car(
                    agent_id=agent_id,
                    name=f"Racer {len(self.cars) + 1}",
                    color=color,
                    position=[track_pos["x"], track_pos["y"]],
                    rotation=track_pos["rotation"],
                    speed=agent_state["speed"],
                    lap=1,
                    lap_time=agent_state["lap_time"],
                    track_position=track_pos["track_progress"]
                )
            else:
                car = self.cars[agent_id]
                car.position = [track_pos["x"], track_pos["y"]]
                car.rotation = track_pos["rotation"]
                car.speed = agent_state["speed"]
                car.lap_time = agent_state["lap_time"]
                car.track_position = track_pos["track_progress"]
            
            car = self.cars[agent_id]
            cars_data.append({
                "id": agent_id,
                "name": car.name,
                "color": car.color,
                "x": car.position[0],
                "y": car.position[1],
                "rotation": car.rotation,
                "speed": car.speed,
                "lap": car.lap,
                "lap_time": car.lap_time,
                "track_position": car.track_position
            })
        
        return {
            "cars": cars_data,
            "frame_id": frame_id,
            "timestamp": frame["timestamp"],
            "total_frames": len(self.race_data["frames"]),
            "playing": self.playing,
            "playback_speed": self.playback_speed,
            "track": {
                "center_x": self.track_center_x,
                "center_y": self.track_center_y,
                "width": self.track_width,
                "height": self.track_height
            }
        }
    
    def play(self) -> None:
        """Start playback"""
        self.playing = True
        logger.info("Started playback")
    
    def pause(self) -> None:
        """Pause playback"""
        self.playing = False
        logger.info("Paused playback")
    
    def set_speed(self, speed: float) -> None:
        """Set playback speed"""
        self.playback_speed = speed
        logger.info(f"Set playback speed to {speed}x")
    
    def set_frame(self, frame_id: int) -> None:
        """Jump to specific frame"""
        if self.race_data:
            self.current_frame = max(0, min(frame_id, len(self.race_data["frames"]) - 1))
            logger.info(f"Set frame to {self.current_frame}")

# Global instances
viewer = TrackManiaViewer()

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# FastAPI app
app = FastAPI(title="TrackMania Racing Viewer", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = Path("./static")
static_dir.mkdir(exist_ok=True)

# Serve static files  
app.mount("/static", StaticFiles(directory="./static"), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            if viewer.race_data and viewer.playing:
                frame_data = viewer.get_frame_data(viewer.current_frame)
                await websocket.send_json(frame_data)
                
                # Advance frame
                viewer.current_frame += 1
                if viewer.current_frame >= len(viewer.race_data["frames"]):
                    viewer.current_frame = 0  # Loop
                
                await asyncio.sleep(0.05 / viewer.playback_speed)  # Adjust for speed
            else:
                # Send current frame even when paused
                if viewer.race_data:
                    frame_data = viewer.get_frame_data(viewer.current_frame)
                    await websocket.send_json(frame_data)
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/races")
async def get_races():
    """Get list of available races"""
    races = []
    for race_file in viewer.data_dir.glob("race_*.json"):
        try:
            with open(race_file, 'r') as f:
                data = json.load(f)
            
            race_id = race_file.stem.replace("race_", "")
            races.append({
                "id": race_id,
                "frames": len(data.get("frames", [])),
                "duration": len(data.get("frames", [])) * 0.05,
                "recorded_at": data.get("metadata", {}).get("created_at", "Unknown")
            })
        except:
            continue
    
    return {"races": races}

@app.post("/api/load/{race_id}")
async def load_race(race_id: str):
    """Load specific race"""
    if viewer.load_race(race_id):
        return {"status": "loaded", "race_id": race_id}
    else:
        raise HTTPException(status_code=404, detail="Race not found")

@app.post("/api/play")
async def play_race():
    """Start playback"""
    viewer.play()
    return {"status": "playing"}

@app.post("/api/pause")
async def pause_race():
    """Pause playback"""
    viewer.pause()
    return {"status": "paused"}

@app.post("/api/speed/{speed}")
async def set_speed(speed: float):
    """Set playback speed"""
    viewer.set_speed(speed)
    return {"status": "speed_set", "speed": speed}

@app.post("/api/frame/{frame_id}")
async def set_frame(frame_id: int):
    """Jump to frame"""
    viewer.set_frame(frame_id)
    return {"status": "frame_set", "frame": frame_id}

@app.get("/")
async def root():
    """Main racing viewer interface"""
    return HTMLResponse(open("/Users/mac/Desktop/new/trackmania-RL/viewer/trackmania_template.html").read())

if __name__ == "__main__":
    logger.info("Starting TrackMania Racing Viewer...")
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")