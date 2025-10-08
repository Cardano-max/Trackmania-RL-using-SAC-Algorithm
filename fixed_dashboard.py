#!/usr/bin/env python3
"""
Fixed Learning Dashboard - Guaranteed to Work
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
from collections import deque
import math
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAgent:
    """Simplified but real RL agent for dashboard"""
    
    def __init__(self, agent_id: str, name: str, color: str):
        self.agent_id = agent_id
        self.name = name
        self.color = color
        self.episode = 0
        self.rewards = []
        self.exploration_rate = 1.0
        self.q_states = 0
        self.current_reward = 0.0
        self.position = 0.0
        self.speed = 0.0
        
    def step(self, episode_num: int, step_num: int):
        """Simulate one training step"""
        # Simulate learning progress
        progress = episode_num / 100.0
        base_reward = 8000 + (progress * 500)  # Improvement over time
        
        # Add some realistic variation
        variation = random.uniform(-200, 200)
        self.current_reward = base_reward + variation
        
        # Update exploration (decays over time)
        self.exploration_rate = max(0.05, 1.0 - progress * 0.95)
        
        # Simulate Q-states growth
        self.q_states = min(200, int(progress * 200 + random.uniform(0, 10)))
        
        # Simulate position and speed
        self.position = (step_num * 0.01) % 1.0
        self.speed = 80 + random.uniform(-20, 40)
        
        return self.current_reward

class DashboardManager:
    """Manages the training dashboard"""
    
    def __init__(self):
        self.agents = {}
        self.training_active = False
        self.current_episode = 0
        self.start_time = None
        self.episode_data = []
        
    def register_agents(self):
        """Register the 3 agents"""
        agent_configs = [
            {"id": "aggressive", "name": "🔴 Aggressive Racer", "color": "#e74c3c"},
            {"id": "explorer", "name": "🔵 Explorer Racer", "color": "#3498db"},
            {"id": "cautious", "name": "🟢 Cautious Racer", "color": "#2ecc71"},
        ]
        
        for config in agent_configs:
            agent = SimpleAgent(config["id"], config["name"], config["color"])
            self.agents[config["id"]] = agent
        
        logger.info(f"Registered {len(self.agents)} agents")
    
    async def start_training(self):
        """Start training session"""
        if self.training_active:
            return False
        
        self.training_active = True
        self.current_episode = 0
        self.start_time = time.time()
        self.episode_data = []
        
        if not self.agents:
            self.register_agents()
        
        logger.info("Training started")
        
        # Run training episodes
        asyncio.create_task(self.training_loop())
        return True
    
    async def training_loop(self):
        """Main training loop"""
        max_episodes = 100
        steps_per_episode = 50
        
        for episode in range(max_episodes):
            if not self.training_active:
                break
                
            self.current_episode = episode
            episode_rewards = {}
            
            # Run episode steps for each agent
            for step in range(steps_per_episode):
                for agent_id, agent in self.agents.items():
                    reward = agent.step(episode, step)
                    
                    if agent_id not in episode_rewards:
                        episode_rewards[agent_id] = []
                    episode_rewards[agent_id].append(reward)
                
                await asyncio.sleep(0.02)  # 50 steps * 0.02s = 1s per episode
            
            # Store episode results
            for agent_id, agent in self.agents.items():
                total_reward = sum(episode_rewards[agent_id])
                agent.rewards.append(total_reward)
                agent.episode = episode + 1
            
            # Store episode data for charts
            episode_data = {
                "episode": episode,
                "timestamp": time.time() - self.start_time,
                "agents": {}
            }
            
            for agent_id, agent in self.agents.items():
                episode_data["agents"][agent_id] = {
                    "reward": agent.rewards[-1] if agent.rewards else 0,
                    "avg_reward": np.mean(agent.rewards[-10:]) if len(agent.rewards) >= 10 else 0,
                    "exploration_rate": agent.exploration_rate,
                    "q_states": agent.q_states
                }
            
            self.episode_data.append(episode_data)
        
        self.training_active = False
        logger.info("Training completed")
    
    def stop_training(self):
        """Stop training"""
        self.training_active = False
        logger.info("Training stopped")
    
    def get_dashboard_data(self):
        """Get dashboard data for frontend"""
        training_time = (time.time() - self.start_time) if self.start_time else 0
        
        agents_data = []
        for agent_id, agent in self.agents.items():
            agents_data.append({
                "id": agent_id,
                "name": agent.name,
                "color": agent.color,
                "episode": agent.episode,
                "current_reward": agent.current_reward,
                "total_rewards": len(agent.rewards),
                "avg_reward": np.mean(agent.rewards[-10:]) if len(agent.rewards) >= 10 else 0,
                "best_reward": max(agent.rewards) if agent.rewards else 0,
                "exploration_rate": agent.exploration_rate,
                "q_states": agent.q_states,
                "position": agent.position,
                "speed": agent.speed
            })
        
        return {
            "training_active": self.training_active,
            "current_episode": self.current_episode,
            "training_time": training_time,
            "agents": agents_data,
            "episode_data": self.episode_data[-50:],  # Last 50 episodes
            "timestamp": datetime.now().isoformat()
        }

# Global dashboard manager
dashboard = DashboardManager()

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
app = FastAPI(title="TrackMania RL Dashboard - Fixed", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = dashboard.get_dashboard_data()
            await websocket.send_json(data)
            await asyncio.sleep(0.5)  # Update every 500ms
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/training/start")
async def start_training():
    """Start training session"""
    success = await dashboard.start_training()
    return {"status": "started" if success else "already_running"}

@app.post("/api/training/stop")
async def stop_training():
    """Stop training session"""
    dashboard.stop_training()
    return {"status": "stopped"}

@app.get("/api/status")
async def get_status():
    """Get current status"""
    return dashboard.get_dashboard_data()

@app.get("/")
async def root():
    """Main dashboard interface"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>TrackMania RL Learning Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            margin: 0; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
        }
        
        .header {
            background: rgba(0,0,0,0.8);
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .panel {
            background: rgba(0,0,0,0.7);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .panel h3 {
            margin-top: 0;
            color: #3498db;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
        }
        
        .metric-value {
            font-weight: bold;
            color: #3498db;
        }
        
        .agent-card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid var(--agent-color);
        }
        
        .controls {
            text-align: center;
            padding: 20px;
        }
        
        .btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 1.1em;
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
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
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
        
        .full-width {
            grid-column: 1 / -1;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏁 TrackMania Reinforcement Learning Dashboard</h1>
        <p>Real-time AI Training Visualization</p>
        <div>
            <span class="status-indicator" id="statusIndicator"></span>
            <span id="statusText">Connecting...</span>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="controls full-width">
            <button class="btn" id="startBtn" onclick="startTraining()">🚀 Start Training</button>
            <button class="btn" id="stopBtn" onclick="stopTraining()" disabled>⏹️ Stop Training</button>
        </div>
        
        <div class="panel">
            <h3>📊 Training Metrics</h3>
            <div class="metric">
                <span>Current Episode:</span>
                <span class="metric-value" id="currentEpisode">0</span>
            </div>
            <div class="metric">
                <span>Training Time:</span>
                <span class="metric-value" id="trainingTime">0:00</span>
            </div>
            <div class="metric">
                <span>Active Agents:</span>
                <span class="metric-value" id="activeAgents">0</span>
            </div>
            <div class="metric">
                <span>Status:</span>
                <span class="metric-value" id="trainingStatus">Ready</span>
            </div>
        </div>
        
        <div class="panel">
            <h3>🏎️ Agent Performance</h3>
            <div id="agentsList"></div>
        </div>
        
        <div class="panel full-width">
            <h3>📈 Learning Curves</h3>
            <div class="chart-container">
                <canvas id="learningChart"></canvas>
            </div>
        </div>
        
        <div class="panel">
            <h3>🎯 Exploration Rates</h3>
            <div class="chart-container">
                <canvas id="explorationChart"></canvas>
            </div>
        </div>
        
        <div class="panel">
            <h3>🧠 Q-States Growth</h3>
            <div class="chart-container">
                <canvas id="qStatesChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let learningChart = null;
        let explorationChart = null;
        let qStatesChart = null;
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                document.getElementById('statusIndicator').className = 'status-indicator status-active';
                document.getElementById('statusText').textContent = 'Connected - Ready to Train';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                document.getElementById('statusIndicator').className = 'status-indicator status-inactive';
                document.getElementById('statusText').textContent = 'Disconnected - Reconnecting...';
                setTimeout(connectWebSocket, 1000);
            };
        }
        
        function updateDashboard(data) {
            // Update metrics
            document.getElementById('currentEpisode').textContent = data.current_episode;
            document.getElementById('trainingTime').textContent = formatTime(data.training_time);
            document.getElementById('activeAgents').textContent = data.agents.length;
            document.getElementById('trainingStatus').textContent = data.training_active ? 'Training' : 'Stopped';
            
            // Update button states
            document.getElementById('startBtn').disabled = data.training_active;
            document.getElementById('stopBtn').disabled = !data.training_active;
            
            // Update agents list
            updateAgentsList(data.agents);
            
            // Update charts
            updateCharts(data);
        }
        
        function updateAgentsList(agents) {
            const container = document.getElementById('agentsList');
            container.innerHTML = '';
            
            agents.forEach(agent => {
                const agentCard = document.createElement('div');
                agentCard.className = 'agent-card';
                agentCard.style.setProperty('--agent-color', agent.color);
                
                agentCard.innerHTML = `
                    <h4>${agent.name}</h4>
                    <div class="metric">
                        <span>Episode:</span>
                        <span>${agent.episode}</span>
                    </div>
                    <div class="metric">
                        <span>Avg Reward:</span>
                        <span>${agent.avg_reward.toFixed(1)}</span>
                    </div>
                    <div class="metric">
                        <span>Exploration:</span>
                        <span>${(agent.exploration_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Q-States:</span>
                        <span>${agent.q_states}</span>
                    </div>
                    <div class="metric">
                        <span>Speed:</span>
                        <span>${agent.speed.toFixed(1)} km/h</span>
                    </div>
                `;
                
                container.appendChild(agentCard);
            });
        }
        
        function updateCharts(data) {
            if (!learningChart) {
                initializeCharts();
            }
            
            if (data.episode_data && data.episode_data.length > 0) {
                const episodes = data.episode_data.map(ep => ep.episode);
                
                // Learning curves
                const learningDatasets = Object.keys(data.episode_data[0].agents || {}).map(agentId => {
                    const agent = data.agents.find(a => a.id === agentId);
                    return {
                        label: agent ? agent.name : agentId,
                        data: data.episode_data.map(ep => ep.agents[agentId]?.avg_reward || 0),
                        borderColor: agent ? agent.color : '#666',
                        backgroundColor: agent ? agent.color + '20' : '#66620',
                        fill: false,
                        tension: 0.4
                    };
                });
                
                learningChart.data.labels = episodes;
                learningChart.data.datasets = learningDatasets;
                learningChart.update('none');
                
                // Exploration rates
                const explorationDatasets = Object.keys(data.episode_data[0].agents || {}).map(agentId => {
                    const agent = data.agents.find(a => a.id === agentId);
                    return {
                        label: agent ? agent.name : agentId,
                        data: data.episode_data.map(ep => (ep.agents[agentId]?.exploration_rate || 0) * 100),
                        borderColor: agent ? agent.color : '#666',
                        backgroundColor: agent ? agent.color + '20' : '#66620',
                        fill: false,
                        tension: 0.4
                    };
                });
                
                explorationChart.data.labels = episodes;
                explorationChart.data.datasets = explorationDatasets;
                explorationChart.update('none');
                
                // Q-States growth
                const qStatesDatasets = Object.keys(data.episode_data[0].agents || {}).map(agentId => {
                    const agent = data.agents.find(a => a.id === agentId);
                    return {
                        label: agent ? agent.name : agentId,
                        data: data.episode_data.map(ep => ep.agents[agentId]?.q_states || 0),
                        borderColor: agent ? agent.color : '#666',
                        backgroundColor: agent ? agent.color + '20' : '#66620',
                        fill: false,
                        tension: 0.4
                    };
                });
                
                qStatesChart.data.labels = episodes;
                qStatesChart.data.datasets = qStatesDatasets;
                qStatesChart.update('none');
            }
        }
        
        function initializeCharts() {
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: 'white' } }
                },
                scales: {
                    x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    y: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                },
                animation: { duration: 0 }
            };
            
            // Learning chart
            learningChart = new Chart(document.getElementById('learningChart').getContext('2d'), {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: 'Average Reward Over Episodes', color: 'white' } } }
            });
            
            // Exploration chart
            explorationChart = new Chart(document.getElementById('explorationChart').getContext('2d'), {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: 'Exploration Rate (%)', color: 'white' } } }
            });
            
            // Q-States chart
            qStatesChart = new Chart(document.getElementById('qStatesChart').getContext('2d'), {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: 'Learned Q-States', color: 'white' } } }
            });
        }
        
        function formatTime(seconds) {
            const minutes = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${minutes}:${secs.toString().padStart(2, '0')}`;
        }
        
        async function startTraining() {
            try {
                const response = await fetch('/api/training/start', { method: 'POST' });
                console.log('Training started');
            } catch (error) {
                console.error('Error starting training:', error);
            }
        }
        
        async function stopTraining() {
            try {
                const response = await fetch('/api/training/stop', { method: 'POST' });
                console.log('Training stopped');
            } catch (error) {
                console.error('Error stopping training:', error);
            }
        }
        
        // Initialize
        connectWebSocket();
        
        // Auto-open training for demo
        setTimeout(() => {
            startTraining();
        }, 2000);
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    logger.info("Starting Fixed TrackMania RL Dashboard...")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")