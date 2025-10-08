#!/usr/bin/env python3
"""
3D TrackMania Racing Visualization
Complete with 3D cars, track, and realistic racing physics
"""

import asyncio
import json
import time
import math
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Track3D:
    """3D Track generator with elevation and banking"""
    
    def __init__(self):
        self.waypoints = []
        self.generate_track()
    
    def generate_track(self):
        """Generate realistic 3D track with elevation changes"""
        num_points = 200
        
        for i in range(num_points):
            progress = i / num_points
            angle = progress * 2 * math.pi
            
            # Base track shape - figure-8 with elevation
            if progress < 0.25:  # Start straight with banking
                x = progress * 800 - 400
                y = 0
                z = 0
                banking = 0
                speed_limit = 200
            elif progress < 0.5:  # Right turn with elevation
                local_progress = (progress - 0.25) * 4
                turn_angle = local_progress * math.pi
                radius = 150
                x = 400 + radius * math.cos(turn_angle - math.pi/2)
                y = radius * math.sin(turn_angle - math.pi/2)
                z = 50 * math.sin(local_progress * math.pi)  # Hill
                banking = -15 * math.sin(local_progress * math.pi)  # Banking
                speed_limit = 120
            elif progress < 0.75:  # Left chicane section
                local_progress = (progress - 0.5) * 4
                x = 400 - local_progress * 800
                y = 150 + 100 * math.sin(local_progress * 3 * math.pi)
                z = 50 - local_progress * 30
                banking = 10 * math.sin(local_progress * 2 * math.pi)
                speed_limit = 160
            else:  # Final turn back to start
                local_progress = (progress - 0.75) * 4
                turn_angle = local_progress * math.pi
                radius = 200
                x = -400 + radius * math.cos(turn_angle + math.pi/2)
                y = radius * math.sin(turn_angle + math.pi/2)
                z = 20 - local_progress * 20
                banking = 8
                speed_limit = 180
            
            # Calculate track direction for car rotation
            next_i = (i + 1) % num_points
            next_progress = next_i / num_points
            next_angle = next_progress * 2 * math.pi
            
            if next_progress < 0.25:
                next_x = next_progress * 800 - 400
                next_y = 0
            elif next_progress < 0.5:
                local_next = (next_progress - 0.25) * 4
                next_angle_turn = local_next * math.pi
                next_x = 400 + 150 * math.cos(next_angle_turn - math.pi/2)
                next_y = 150 * math.sin(next_angle_turn - math.pi/2)
            elif next_progress < 0.75:
                local_next = (next_progress - 0.5) * 4
                next_x = 400 - local_next * 800
                next_y = 150 + 100 * math.sin(local_next * 3 * math.pi)
            else:
                local_next = (next_progress - 0.75) * 4
                next_angle_turn = local_next * math.pi
                next_x = -400 + 200 * math.cos(next_angle_turn + math.pi/2)
                next_y = 200 * math.sin(next_angle_turn + math.pi/2)
            
            # Calculate heading
            heading = math.atan2(next_y - y, next_x - x)
            
            self.waypoints.append({
                'x': x,
                'y': y, 
                'z': z,
                'heading': heading,
                'banking': banking,
                'speed_limit': speed_limit,
                'sector': int(progress * 4),  # 4 sectors
                'checkpoint': i % 20 == 0  # Checkpoints every 20 points
            })

class RacingCar:
    """3D Racing car with realistic physics"""
    
    def __init__(self, car_id: str, name: str, color: str, track: Track3D):
        self.car_id = car_id
        self.name = name
        self.color = color
        self.track = track
        
        # Position and motion
        self.track_position = random.uniform(0, 1)  # 0-1 around track
        self.speed = 0.0  # km/h
        self.acceleration = 0.0
        self.lateral_offset = random.uniform(-5, 5)  # Distance from racing line
        
        # Car state
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        self.gear = 1
        self.rpm = 1000
        
        # Racing data
        self.lap_time = 0.0
        self.sector_times = [0.0, 0.0, 0.0, 0.0]
        self.current_sector = 0
        self.lap_count = 0
        self.best_lap = float('inf')
        
        # Physics
        self.max_speed = 250 + random.uniform(-20, 30)
        self.acceleration_rate = 8.0 + random.uniform(-1, 2)
        self.braking_rate = 12.0 + random.uniform(-2, 2)
        
        # AI behavior
        self.aggressiveness = random.uniform(0.3, 1.0)
        self.skill_level = random.uniform(0.6, 0.95)
        
    def get_world_position(self):
        """Get 3D world position of car"""
        track_index = int(self.track_position * len(self.track.waypoints)) % len(self.track.waypoints)
        waypoint = self.track.waypoints[track_index]
        
        # Calculate position with lateral offset
        heading = waypoint['heading']
        offset_x = self.lateral_offset * math.cos(heading + math.pi/2)
        offset_y = self.lateral_offset * math.sin(heading + math.pi/2)
        
        return {
            'x': waypoint['x'] + offset_x,
            'y': waypoint['y'] + offset_y,
            'z': waypoint['z'] + 2.0,  # Car height above track
            'heading': heading + self.steering * 0.3,
            'banking': waypoint['banking']
        }
    
    def update_physics(self, dt: float = 0.05):
        """Update car physics"""
        track_index = int(self.track_position * len(self.track.waypoints)) % len(self.track.waypoints)
        waypoint = self.track.waypoints[track_index]
        
        # AI driving behavior
        target_speed = waypoint['speed_limit'] * self.skill_level
        speed_diff = target_speed - self.speed
        
        # Throttle/brake decision
        if speed_diff > 10:
            self.throttle = min(1.0, self.aggressiveness)
            self.brake = 0.0
        elif speed_diff < -20:
            self.throttle = 0.0
            self.brake = min(1.0, abs(speed_diff) / 50.0)
        else:
            self.throttle = 0.6
            self.brake = 0.0
        
        # Steering for racing line
        ideal_offset = 0.0  # Racing line center
        if waypoint['banking'] != 0:
            ideal_offset = -waypoint['banking'] * 0.3  # Use banking
        
        offset_error = self.lateral_offset - ideal_offset
        self.steering = -offset_error * 0.1 * self.skill_level
        self.steering = max(-1.0, min(1.0, self.steering))
        
        # Update lateral position
        self.lateral_offset += self.steering * self.speed * dt * 0.01
        self.lateral_offset = max(-15, min(15, self.lateral_offset))  # Track limits
        
        # Speed physics
        engine_force = self.throttle * self.acceleration_rate
        brake_force = self.brake * self.braking_rate
        air_resistance = 0.002 * self.speed * self.speed
        
        net_force = engine_force - brake_force - air_resistance
        self.acceleration = net_force
        
        # Update speed
        self.speed += self.acceleration * dt
        self.speed = max(0, min(self.max_speed, self.speed))
        
        # Update track position
        track_length = 3200  # Approximate track length in meters
        speed_ms = self.speed / 3.6  # Convert km/h to m/s
        position_delta = (speed_ms * dt) / track_length
        
        old_position = self.track_position
        self.track_position = (self.track_position + position_delta) % 1.0
        
        # Check for lap completion
        if old_position > 0.9 and self.track_position < 0.1:
            self.lap_count += 1
            if self.lap_time > 0:
                self.best_lap = min(self.best_lap, self.lap_time)
            self.lap_time = 0.0
        
        # Update sector times
        new_sector = int(self.track_position * 4)
        if new_sector != self.current_sector:
            self.current_sector = new_sector
        
        # Update lap time
        self.lap_time += dt
        
        # Update RPM and gear
        self.rpm = 1000 + self.speed * 50 + random.uniform(-100, 100)
        self.gear = min(6, max(1, int(self.speed / 40) + 1))
        
        return self.get_world_position()

class Racing3DManager:
    """Manages 3D racing simulation"""
    
    def __init__(self):
        self.track = Track3D()
        self.cars = {}
        self.simulation_active = False
        self.start_time = None
        self.race_time = 0.0
        
    def add_car(self, car_id: str, name: str, color: str):
        """Add a racing car"""
        car = RacingCar(car_id, name, color, self.track)
        # Stagger starting positions
        car.track_position = len(self.cars) * 0.02
        self.cars[car_id] = car
        logger.info(f"Added car: {name}")
    
    def start_race(self):
        """Start the 3D race simulation"""
        if self.simulation_active:
            return False
        
        self.simulation_active = True
        self.start_time = time.time()
        self.race_time = 0.0
        
        # Add 3 AI cars if none exist
        if not self.cars:
            self.add_car("car1", "Lightning McQueen", "#ff4444")
            self.add_car("car2", "Speed Racer", "#4444ff") 
            self.add_car("car3", "Green Machine", "#44ff44")
        
        # Start simulation loop
        asyncio.create_task(self.simulation_loop())
        logger.info("3D Race started")
        return True
    
    def stop_race(self):
        """Stop the race simulation"""
        self.simulation_active = False
        logger.info("Race stopped")
    
    async def simulation_loop(self):
        """Main simulation loop"""
        while self.simulation_active:
            dt = 0.05  # 20 FPS
            self.race_time = time.time() - self.start_time if self.start_time else 0
            
            # Update all cars
            for car in self.cars.values():
                car.update_physics(dt)
            
            await asyncio.sleep(dt)
    
    def get_simulation_data(self):
        """Get complete 3D simulation data"""
        cars_data = []
        
        for car in self.cars.values():
            position = car.get_world_position()
            
            cars_data.append({
                'id': car.car_id,
                'name': car.name,
                'color': car.color,
                'position': position,
                'speed': car.speed,
                'throttle': car.throttle,
                'brake': car.brake,
                'steering': car.steering,
                'gear': car.gear,
                'rpm': car.rpm,
                'lap_count': car.lap_count,
                'lap_time': car.lap_time,
                'best_lap': car.best_lap if car.best_lap != float('inf') else 0,
                'sector': car.current_sector,
                'track_position': car.track_position
            })
        
        # Sort by track position for race order
        cars_data.sort(key=lambda x: x['track_position'], reverse=True)
        for i, car in enumerate(cars_data):
            car['position_rank'] = i + 1
        
        return {
            'race_active': self.simulation_active,
            'race_time': self.race_time,
            'cars': cars_data,
            'track': self.get_track_data(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_track_data(self):
        """Get track data for 3D rendering"""
        return [
            {
                'x': wp['x'],
                'y': wp['y'],
                'z': wp['z'],
                'heading': wp['heading'],
                'banking': wp['banking'],
                'speed_limit': wp['speed_limit'],
                'checkpoint': wp['checkpoint']
            }
            for wp in self.track.waypoints
        ]

# Global racing manager
racing_manager = Racing3DManager()

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# FastAPI app
app = FastAPI(title="TrackMania 3D Racing Viewer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time 3D data"""
    await manager.connect(websocket)
    try:
        while True:
            data = racing_manager.get_simulation_data()
            await websocket.send_json(data)
            await asyncio.sleep(0.05)  # 20 FPS
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/race/start")
async def start_race():
    """Start 3D race"""
    success = racing_manager.start_race()
    return {"status": "started" if success else "already_running"}

@app.post("/api/race/stop")
async def stop_race():
    """Stop 3D race"""
    racing_manager.stop_race()
    return {"status": "stopped"}

@app.get("/api/status")
async def get_status():
    """Get race status"""
    return racing_manager.get_simulation_data()

@app.get("/")
async def root():
    """3D Racing viewer interface"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>TrackMania 3D Racing Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body { 
            margin: 0; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: #000; 
            color: white; 
            overflow: hidden;
        }
        
        #container {
            position: relative;
            width: 100vw;
            height: 100vh;
        }
        
        #racing-canvas {
            display: block;
        }
        
        .hud {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.8);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            min-width: 300px;
            z-index: 100;
        }
        
        .controls {
            position: absolute;
            bottom: 20px;
            left: 20px;
            z-index: 100;
        }
        
        .btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        
        .btn:disabled {
            background: #7f8c8d;
            cursor: not-allowed;
            transform: none;
        }
        
        .leaderboard {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            min-width: 250px;
            z-index: 100;
        }
        
        .car-info {
            display: flex;
            align-items: center;
            margin: 8px 0;
            padding: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }
        
        .car-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid rgba(255,255,255,0.5);
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }
        
        .metric-value {
            font-weight: bold;
            color: #3498db;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .status-active { background: #2ecc71; animation: pulse 2s infinite; }
        .status-inactive { background: #e74c3c; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .instructions {
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 10px;
            font-size: 0.9em;
            max-width: 200px;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="container">
        <canvas id="racing-canvas"></canvas>
        
        <div class="hud">
            <h3>🏁 TrackMania 3D Racing</h3>
            <div class="metric">
                <span>Status:</span>
                <span>
                    <span class="status-indicator" id="statusIndicator"></span>
                    <span id="raceStatus">Ready</span>
                </span>
            </div>
            <div class="metric">
                <span>Race Time:</span>
                <span class="metric-value" id="raceTime">0:00</span>
            </div>
            <div class="metric">
                <span>Active Cars:</span>
                <span class="metric-value" id="activeCars">0</span>
            </div>
            <div class="metric">
                <span>Camera:</span>
                <span class="metric-value">Mouse to rotate</span>
            </div>
        </div>
        
        <div class="leaderboard">
            <h4>🏆 Live Leaderboard</h4>
            <div id="leaderboardList"></div>
        </div>
        
        <div class="controls">
            <button class="btn" id="startBtn" onclick="startRace()">🚀 Start Race</button>
            <button class="btn" id="stopBtn" onclick="stopRace()" disabled>⏹️ Stop Race</button>
            <button class="btn" onclick="resetCamera()">📷 Reset Camera</button>
        </div>
        
        <div class="instructions">
            <strong>🎮 Controls:</strong><br>
            • Mouse: Rotate camera<br>
            • Scroll: Zoom in/out<br>
            • Right-click: Pan view<br>
            <br>
            <strong>🏎️ Features:</strong><br>
            • Real 3D physics<br>
            • Dynamic racing AI<br>
            • Live telemetry<br>
        </div>
    </div>

    <script>
        let scene, camera, renderer, controls;
        let cars = {};
        let trackMesh;
        let ws = null;
        
        // Initialize 3D scene
        function init3DScene() {
            const container = document.getElementById('container');
            const canvas = document.getElementById('racing-canvas');
            
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x87CEEB); // Sky blue
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 10000);
            camera.position.set(0, 500, 800);
            camera.lookAt(0, 0, 0);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(100, 200, 50);
            directionalLight.castShadow = true;
            directionalLight.shadow.mapSize.width = 2048;
            directionalLight.shadow.mapSize.height = 2048;
            scene.add(directionalLight);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxPolarAngle = Math.PI / 2.2;
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize, false);
            
            // Start animation loop
            animate();
        }
        
        function createTrack(trackData) {
            if (trackMesh) {
                scene.remove(trackMesh);
            }
            
            const trackGeometry = new THREE.BufferGeometry();
            const positions = [];
            const colors = [];
            
            // Create track surface
            for (let i = 0; i < trackData.length - 1; i++) {
                const current = trackData[i];
                const next = trackData[i + 1];
                
                const trackWidth = 20;
                
                // Calculate perpendicular direction for track width
                const dx = next.x - current.x;
                const dy = next.y - current.y;
                const length = Math.sqrt(dx * dx + dy * dy);
                const perpX = -dy / length * trackWidth;
                const perpY = dx / length * trackWidth;
                
                // Create track quad
                positions.push(
                    current.x - perpX, current.z, current.y - perpY,
                    current.x + perpX, current.z, current.y + perpY,
                    next.x - perpX, next.z, next.y - perpY,
                    next.x + perpX, next.z, next.y + perpY
                );
                
                // Track surface color (darker for racing line)
                const intensity = current.checkpoint ? 0.8 : 0.6;
                for (let j = 0; j < 4; j++) {
                    colors.push(intensity, intensity, intensity);
                }
            }
            
            trackGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            trackGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            const trackMaterial = new THREE.MeshLambertMaterial({ 
                vertexColors: true,
                side: THREE.DoubleSide 
            });
            
            trackMesh = new THREE.Mesh(trackGeometry, trackMaterial);
            trackMesh.receiveShadow = true;
            scene.add(trackMesh);
            
            // Add track barriers
            addTrackBarriers(trackData);
        }
        
        function addTrackBarriers(trackData) {
            const barrierGeometry = new THREE.BoxGeometry(1, 10, 1);
            const barrierMaterial = new THREE.MeshLambertMaterial({ color: 0xff4444 });
            
            for (let i = 0; i < trackData.length; i += 10) {
                const point = trackData[i];
                
                // Left barrier
                const leftBarrier = new THREE.Mesh(barrierGeometry, barrierMaterial);
                leftBarrier.position.set(point.x - 25, point.z + 5, point.y);
                scene.add(leftBarrier);
                
                // Right barrier
                const rightBarrier = new THREE.Mesh(barrierGeometry, barrierMaterial);
                rightBarrier.position.set(point.x + 25, point.z + 5, point.y);
                scene.add(rightBarrier);
            }
        }
        
        function createCar(carData) {
            const carGroup = new THREE.Group();
            
            // Car body
            const bodyGeometry = new THREE.BoxGeometry(8, 3, 16);
            const bodyMaterial = new THREE.MeshLambertMaterial({ color: carData.color });
            const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
            body.position.y = 1.5;
            body.castShadow = true;
            carGroup.add(body);
            
            // Car windshield
            const windshieldGeometry = new THREE.BoxGeometry(6, 2, 4);
            const windshieldMaterial = new THREE.MeshLambertMaterial({ 
                color: 0x4444ff, 
                transparent: true, 
                opacity: 0.7 
            });
            const windshield = new THREE.Mesh(windshieldGeometry, windshieldMaterial);
            windshield.position.set(0, 3, 3);
            carGroup.add(windshield);
            
            // Wheels
            const wheelGeometry = new THREE.CylinderGeometry(1.5, 1.5, 1, 8);
            const wheelMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 });
            
            const wheels = [];
            const wheelPositions = [
                [-3, 0, -5], [3, 0, -5], [-3, 0, 5], [3, 0, 5]
            ];
            
            wheelPositions.forEach(pos => {
                const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
                wheel.position.set(pos[0], pos[1], pos[2]);
                wheel.rotation.z = Math.PI / 2;
                wheel.castShadow = true;
                carGroup.add(wheel);
                wheels.push(wheel);
            });
            
            // Store wheels for animation
            carGroup.userData.wheels = wheels;
            carGroup.userData.carData = carData;
            
            return carGroup;
        }
        
        function updateCars(carsData) {
            carsData.forEach(carData => {
                if (!cars[carData.id]) {
                    cars[carData.id] = createCar(carData);
                    scene.add(cars[carData.id]);
                }
                
                const carMesh = cars[carData.id];
                const pos = carData.position;
                
                // Update position
                carMesh.position.set(pos.x, pos.z + 2, pos.y);
                carMesh.rotation.y = -pos.heading;
                carMesh.rotation.z = pos.banking * Math.PI / 180;
                
                // Animate wheels based on speed
                const wheelSpeed = carData.speed * 0.1;
                carMesh.userData.wheels.forEach(wheel => {
                    wheel.rotation.x += wheelSpeed;
                });
                
                // Update steering on front wheels
                const steeringAngle = carData.steering * 0.3;
                carMesh.userData.wheels[0].rotation.y = steeringAngle; // Front left
                carMesh.userData.wheels[1].rotation.y = steeringAngle; // Front right
            });
        }
        
        function updateHUD(data) {
            // Update status
            document.getElementById('raceStatus').textContent = data.race_active ? 'Racing' : 'Stopped';
            document.getElementById('statusIndicator').className = 
                'status-indicator ' + (data.race_active ? 'status-active' : 'status-inactive');
            
            // Update race time
            const minutes = Math.floor(data.race_time / 60);
            const seconds = Math.floor(data.race_time % 60);
            document.getElementById('raceTime').textContent = 
                `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            // Update car count
            document.getElementById('activeCars').textContent = data.cars.length;
            
            // Update buttons
            document.getElementById('startBtn').disabled = data.race_active;
            document.getElementById('stopBtn').disabled = !data.race_active;
            
            // Update leaderboard
            updateLeaderboard(data.cars);
        }
        
        function updateLeaderboard(carsData) {
            const container = document.getElementById('leaderboardList');
            container.innerHTML = '';
            
            // Sort by position rank
            const sortedCars = [...carsData].sort((a, b) => a.position_rank - b.position_rank);
            
            sortedCars.forEach(car => {
                const carInfo = document.createElement('div');
                carInfo.className = 'car-info';
                
                const bestLap = car.best_lap > 0 ? formatTime(car.best_lap) : '--:--';
                
                carInfo.innerHTML = `
                    <div class="car-color" style="background: ${car.color}"></div>
                    <div style="flex: 1;">
                        <div><strong>${car.position_rank}. ${car.name}</strong></div>
                        <div style="font-size: 0.8em; opacity: 0.8;">
                            ${car.speed.toFixed(0)} km/h | Lap ${car.lap_count + 1}
                        </div>
                        <div style="font-size: 0.8em; opacity: 0.8;">
                            Best: ${bestLap}
                        </div>
                    </div>
                `;
                
                container.appendChild(carInfo);
            });
        }
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(1);
            return `${mins}:${secs.padStart(4, '0')}`;
        }
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // Create track on first data
                if (data.track && !trackMesh) {
                    createTrack(data.track);
                }
                
                // Update cars
                if (data.cars) {
                    updateCars(data.cars);
                }
                
                // Update HUD
                updateHUD(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected. Reconnecting...');
                setTimeout(connectWebSocket, 1000);
            };
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            if (controls) {
                controls.update();
            }
            
            renderer.render(scene, camera);
        }
        
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
        
        function resetCamera() {
            camera.position.set(0, 500, 800);
            camera.lookAt(0, 0, 0);
            controls.reset();
        }
        
        async function startRace() {
            try {
                const response = await fetch('/api/race/start', { method: 'POST' });
                console.log('Race started');
            } catch (error) {
                console.error('Error starting race:', error);
            }
        }
        
        async function stopRace() {
            try {
                const response = await fetch('/api/race/stop', { method: 'POST' });
                console.log('Race stopped');
            } catch (error) {
                console.error('Error stopping race:', error);
            }
        }
        
        // Initialize everything
        init3DScene();
        connectWebSocket();
        
        // Auto-start race for demo
        setTimeout(() => {
            startRace();
        }, 2000);
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    logger.info("Starting TrackMania 3D Racing Viewer...")
    uvicorn.run(app, host="0.0.0.0", port=6001, log_level="info")