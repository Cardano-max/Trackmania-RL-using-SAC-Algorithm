#!/usr/bin/env python3
"""
Production RL Learning Dashboard
Shows real-time RL learning metrics and progress
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
from collections import deque

from rl_learning_engine import ProductionRLEnvironment, ActionSpace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningDashboard:
    """Advanced dashboard showing RL learning process"""
    
    def __init__(self):
        self.environment = ProductionRLEnvironment()
        self.active_agents = {}
        self.learning_history = deque(maxlen=1000)
        self.training_active = False
        self.current_episode = 0
        
        # Performance tracking
        self.episodes_completed = 0
        self.total_training_time = 0.0
        self.best_episode_reward = float('-inf')
        self.convergence_threshold = 0.95  # 95% of optimal performance
        
    async def start_training_session(self, num_agents: int = 3, episodes_per_agent: int = 50):
        """Start comprehensive training session"""
        
        self.training_active = True
        self.current_episode = 0
        
        # Register agents with different learning parameters
        agent_configs = [
            {"id": "aggressive_learner", "name": "Aggressive Racer", "color": "#e74c3c"},
            {"id": "conservative_learner", "name": "Conservative Racer", "color": "#3498db"},
            {"id": "balanced_learner", "name": "Balanced Racer", "color": "#2ecc71"},
        ]
        
        for config in agent_configs[:num_agents]:
            agent = self.environment.register_agent(config["id"])
            self.active_agents[config["id"]] = {
                "agent": agent,
                "name": config["name"],
                "color": config["color"],
                "episode_rewards": [],
                "learning_curves": [],
                "exploration_history": [],
                "q_value_history": []
            }
        
        logger.info(f"Started training session with {num_agents} agents for {episodes_per_agent} episodes each")
        
        # Run training episodes
        for episode in range(episodes_per_agent):
            self.current_episode = episode
            
            episode_start_time = time.time()
            
            # Run episode for each agent
            for agent_id, agent_data in self.active_agents.items():
                await self.run_episode(agent_id, episode)
            
            episode_duration = time.time() - episode_start_time
            self.total_training_time += episode_duration
            
            # Log progress
            if episode % 10 == 0:
                self.log_training_progress(episode, episodes_per_agent)
            
            # Small delay for real-time visualization
            await asyncio.sleep(0.1)
        
        self.training_active = False
        logger.info("Training session completed!")
    
    async def run_episode(self, agent_id: str, episode_num: int):
        """Run single episode for agent"""
        
        agent_data = self.active_agents[agent_id]
        agent = agent_data["agent"]
        
        episode_reward = 0.0
        episode_steps = 0
        episode_q_values = []
        
        # Reset environment
        current_state = self.environment.car_states[agent_id]
        
        # Run episode (120 seconds at 20Hz = 2400 steps)
        max_steps = 2400
        
        for step in range(max_steps):
            # Get track information
            track_info = self.environment.track.get_track_info(current_state.track_position)
            
            # Agent decides action
            action = agent.get_action(current_state, track_info)
            
            # Execute action in environment
            next_state, reward, done, metrics = self.environment.step(agent_id, action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Track Q-values
            if agent.q_values:
                episode_q_values.append(agent.q_values[-1])
            
            # Store learning data for visualization
            learning_data = {
                "agent_id": agent_id,
                "episode": episode_num,
                "step": step,
                "reward": reward,
                "cumulative_reward": episode_reward,
                "q_value": episode_q_values[-1] if episode_q_values else 0.0,
                "exploration_rate": metrics.exploration_rate,
                "policy_loss": metrics.policy_loss,
                "track_position": next_state.track_position,
                "speed": next_state.speed,
                "distance_to_line": next_state.distance_to_racing_line,
                "timestamp": time.time()
            }
            self.learning_history.append(learning_data)
            
            current_state = next_state
            
            if done:
                break
            
            # Small delay for real-time effect
            await asyncio.sleep(0.001)
        
        # Episode completed - record results
        agent_data["episode_rewards"].append(episode_reward)
        agent_data["exploration_history"].append(agent.exploration_rate)
        
        # Calculate learning curve metrics
        avg_reward = np.mean(agent_data["episode_rewards"][-10:])
        agent_data["learning_curves"].append({
            "episode": episode_num,
            "reward": episode_reward,
            "average_reward": avg_reward,
            "exploration_rate": agent.exploration_rate,
            "steps": episode_steps
        })
        
        # Update best performance
        if episode_reward > self.best_episode_reward:
            self.best_episode_reward = episode_reward
        
        logger.info(f"Agent {agent_id} Episode {episode_num}: Reward={episode_reward:.2f}, Steps={episode_steps}, Exploration={agent.exploration_rate:.3f}")
    
    def log_training_progress(self, episode: int, total_episodes: int):
        """Log comprehensive training progress"""
        
        progress_pct = (episode / total_episodes) * 100
        
        # Calculate average performance across all agents
        total_rewards = []
        total_exploration = []
        
        for agent_data in self.active_agents.values():
            if agent_data["episode_rewards"]:
                total_rewards.extend(agent_data["episode_rewards"][-10:])  # Last 10 episodes
                total_exploration.append(agent_data["exploration_history"][-1])
        
        avg_reward = np.mean(total_rewards) if total_rewards else 0
        avg_exploration = np.mean(total_exploration) if total_exploration else 0
        
        logger.info(f"Training Progress: {progress_pct:.1f}% | Avg Reward: {avg_reward:.2f} | Exploration: {avg_exploration:.3f}")
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data"""
        
        # Agent performance data
        agents_data = []
        for agent_id, agent_data in self.active_agents.items():
            agent = agent_data["agent"]
            metrics = agent.get_learning_metrics()
            
            agents_data.append({
                "id": agent_id,
                "name": agent_data["name"],
                "color": agent_data["color"],
                "current_reward": agent_data["episode_rewards"][-1] if agent_data["episode_rewards"] else 0,
                "average_reward": np.mean(agent_data["episode_rewards"][-10:]) if agent_data["episode_rewards"] else 0,
                "best_reward": max(agent_data["episode_rewards"]) if agent_data["episode_rewards"] else 0,
                "episodes_completed": len(agent_data["episode_rewards"]),
                "exploration_rate": metrics.exploration_rate,
                "learning_rate": metrics.learning_rate,
                "q_value_loss": metrics.q_value_loss,
                "policy_loss": metrics.policy_loss,
                "replay_buffer_size": metrics.replay_buffer_size,
                "learning_curve": agent_data["learning_curves"][-50:],  # Last 50 episodes
                "current_state": {
                    "position": self.environment.car_states[agent_id].position,
                    "speed": self.environment.car_states[agent_id].speed,
                    "track_position": self.environment.car_states[agent_id].track_position,
                    "distance_to_line": self.environment.car_states[agent_id].distance_to_racing_line
                }
            })
        
        # Learning insights
        insights = self.generate_learning_insights()
        
        # Track data for visualization
        track_data = self.environment.get_track_data()
        
        return {
            "training_active": self.training_active,
            "current_episode": self.current_episode,
            "total_training_time": self.total_training_time,
            "best_episode_reward": self.best_episode_reward,
            "agents": agents_data,
            "learning_insights": insights,
            "track": track_data,
            "recent_learning_data": list(self.learning_history)[-100:],  # Last 100 data points
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_learning_insights(self) -> List[Dict]:
        """Generate AI insights about learning progress"""
        
        insights = []
        
        if not self.active_agents:
            return insights
        
        # Performance comparison
        agent_performances = {}
        for agent_id, agent_data in self.active_agents.items():
            if agent_data["episode_rewards"]:
                avg_performance = np.mean(agent_data["episode_rewards"][-10:])
                agent_performances[agent_id] = avg_performance
        
        if agent_performances:
            best_agent = max(agent_performances, key=agent_performances.get)
            worst_agent = min(agent_performances, key=agent_performances.get)
            
            insights.append({
                "type": "performance_ranking",
                "title": "Performance Analysis",
                "message": f"{self.active_agents[best_agent]['name']} is leading with {agent_performances[best_agent]:.1f} avg reward",
                "data": agent_performances
            })
        
        # Learning convergence analysis
        all_rewards = []
        for agent_data in self.active_agents.values():
            all_rewards.extend(agent_data["episode_rewards"][-20:])
        
        if len(all_rewards) > 10:
            recent_variance = np.var(all_rewards[-10:])
            early_variance = np.var(all_rewards[:10])
            
            if recent_variance < early_variance * 0.5:
                insights.append({
                    "type": "convergence",
                    "title": "Learning Stability",
                    "message": f"Learning is converging! Variance reduced by {((early_variance - recent_variance) / early_variance * 100):.1f}%",
                    "data": {"recent_variance": recent_variance, "early_variance": early_variance}
                })
        
        # Exploration vs exploitation insights
        exploration_rates = [agent_data["exploration_history"][-1] for agent_data in self.active_agents.values() if agent_data["exploration_history"]]
        if exploration_rates:
            avg_exploration = np.mean(exploration_rates)
            
            if avg_exploration > 0.5:
                insights.append({
                    "type": "exploration",
                    "title": "Learning Phase",
                    "message": f"Agents are exploring ({avg_exploration:.1%} exploration rate). Expect variable performance.",
                    "data": {"exploration_rate": avg_exploration}
                })
            elif avg_exploration < 0.1:
                insights.append({
                    "type": "exploitation",
                    "title": "Learning Phase", 
                    "message": f"Agents are exploiting learned policies ({avg_exploration:.1%} exploration). Performance should be stable.",
                    "data": {"exploration_rate": avg_exploration}
                })
        
        return insights

# Global dashboard instance
dashboard = LearningDashboard()

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
        for connection in self.active_connections[:]:  # Copy list to avoid modification during iteration
            try:
                await connection.send_json(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# FastAPI app
app = FastAPI(title="TrackMania RL Learning Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time learning data"""
    await manager.connect(websocket)
    try:
        while True:
            dashboard_data = dashboard.get_dashboard_data()
            await websocket.send_json(dashboard_data)
            await asyncio.sleep(0.5)  # Update every 500ms
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/training/start")
async def start_training():
    """Start training session"""
    if dashboard.training_active:
        return {"status": "already_running"}
    
    # Start training in background
    asyncio.create_task(dashboard.start_training_session(num_agents=3, episodes_per_agent=100))
    
    return {"status": "started", "timestamp": datetime.now().isoformat()}

@app.post("/api/training/stop")
async def stop_training():
    """Stop training session"""
    dashboard.training_active = False
    return {"status": "stopped", "timestamp": datetime.now().isoformat()}

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
            overflow-x: hidden;
        }
        
        .header {
            background: rgba(0,0,0,0.8);
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .panel {
            background: rgba(0,0,0,0.7);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
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
            align-items: center;
            margin: 15px 0;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        
        .metric-value {
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .agent-card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid var(--agent-color);
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .insight {
            background: rgba(52, 152, 219, 0.2);
            border: 1px solid rgba(52, 152, 219, 0.5);
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .insight-title {
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        
        .control-panel {
            grid-column: 1 / -1;
            text-align: center;
            padding: 30px;
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
        
        .track-visualization {
            grid-column: 1 / -1;
            height: 400px;
            background: rgba(0,0,0,0.5);
            border-radius: 15px;
            position: relative;
            overflow: hidden;
        }
        
        #trackCanvas {
            width: 100%;
            height: 100%;
        }
        
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏁 TrackMania Reinforcement Learning Dashboard</h1>
        <p>Production-Level AI Training Visualization</p>
        <div>
            <span class="status-indicator" id="statusIndicator"></span>
            <span id="statusText">Connecting...</span>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="control-panel">
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
                <span class="metric-value" id="trainingTime">0:00:00</span>
            </div>
            <div class="metric">
                <span>Best Reward:</span>
                <span class="metric-value" id="bestReward">0</span>
            </div>
            <div class="metric">
                <span>Active Agents:</span>
                <span class="metric-value" id="activeAgents">0</span>
            </div>
        </div>
        
        <div class="panel">
            <h3>🧠 Learning Progress</h3>
            <div class="chart-container">
                <canvas id="learningChart"></canvas>
            </div>
        </div>
        
        <div class="panel">
            <h3>🎯 Agent Performance</h3>
            <div id="agentsList"></div>
        </div>
        
        <div class="panel">
            <h3>💡 Learning Insights</h3>
            <div id="insightsList"></div>
        </div>
        
        <div class="panel">
            <h3>📈 Exploration vs Exploitation</h3>
            <div class="chart-container">
                <canvas id="explorationChart"></canvas>
            </div>
        </div>
        
        <div class="panel">
            <h3>⚡ Real-time Metrics</h3>
            <div class="chart-container">
                <canvas id="realtimeChart"></canvas>
            </div>
        </div>
        
        <div class="track-visualization">
            <h3 style="position: absolute; top: 10px; left: 20px; z-index: 10;">🏎️ Live Track Visualization</h3>
            <canvas id="trackCanvas"></canvas>
        </div>
    </div>

    <script>
        let ws = null;
        let learningChart = null;
        let explorationChart = null;
        let realtimeChart = null;
        let dashboardData = null;
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                document.getElementById('statusIndicator').className = 'status-indicator status-active';
                document.getElementById('statusText').textContent = 'Connected - Ready to Train';
            };
            
            ws.onmessage = function(event) {
                dashboardData = JSON.parse(event.data);
                updateDashboard(dashboardData);
            };
            
            ws.onclose = function() {
                document.getElementById('statusIndicator').className = 'status-indicator status-inactive';
                document.getElementById('statusText').textContent = 'Disconnected - Reconnecting...';
                setTimeout(connectWebSocket, 1000);
            };
        }
        
        function updateDashboard(data) {
            // Update training metrics
            document.getElementById('currentEpisode').textContent = data.current_episode;
            document.getElementById('trainingTime').textContent = formatTime(data.total_training_time);
            document.getElementById('bestReward').textContent = data.best_episode_reward.toFixed(1);
            document.getElementById('activeAgents').textContent = data.agents.length;
            
            // Update training status
            if (data.training_active) {
                document.getElementById('statusText').textContent = 'Training Active';
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
            } else {
                document.getElementById('statusText').textContent = 'Training Stopped';
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            }
            
            // Update agents list
            updateAgentsList(data.agents);
            
            // Update insights
            updateInsights(data.learning_insights);
            
            // Update charts
            updateCharts(data);
            
            // Update track visualization
            updateTrackVisualization(data);
        }
        
        function updateAgentsList(agents) {
            const container = document.getElementById('agentsList');
            container.innerHTML = '';
            
            agents.forEach(agent => {
                const agentCard = document.createElement('div');
                agentCard.className = 'agent-card';
                agentCard.style.setProperty('--agent-color', agent.color);
                
                const explorationPct = (agent.exploration_rate * 100).toFixed(1);
                const rewardProgress = Math.min(100, (agent.average_reward / 500) * 100);
                
                agentCard.innerHTML = `
                    <h4>${agent.name}</h4>
                    <div class="metric">
                        <span>Episodes:</span>
                        <span>${agent.episodes_completed}</span>
                    </div>
                    <div class="metric">
                        <span>Avg Reward:</span>
                        <span>${agent.average_reward.toFixed(1)}</span>
                    </div>
                    <div class="metric">
                        <span>Exploration:</span>
                        <span>${explorationPct}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${rewardProgress}%"></div>
                    </div>
                `;
                
                container.appendChild(agentCard);
            });
        }
        
        function updateInsights(insights) {
            const container = document.getElementById('insightsList');
            container.innerHTML = '';
            
            if (insights.length === 0) {
                container.innerHTML = '<p style="opacity: 0.7;">Training insights will appear here...</p>';
                return;
            }
            
            insights.forEach(insight => {
                const insightDiv = document.createElement('div');
                insightDiv.className = 'insight';
                insightDiv.innerHTML = `
                    <div class="insight-title">${insight.title}</div>
                    <div>${insight.message}</div>
                `;
                container.appendChild(insightDiv);
            });
        }
        
        function updateCharts(data) {
            if (!learningChart) {
                initializeCharts();
            }
            
            if (data.agents.length > 0) {
                // Learning curve chart
                const episodes = data.agents[0].learning_curve.map(point => point.episode);
                const datasets = data.agents.map(agent => ({
                    label: agent.name,
                    data: agent.learning_curve.map(point => point.average_reward),
                    borderColor: agent.color,
                    backgroundColor: agent.color + '20',
                    fill: false,
                    tension: 0.4
                }));
                
                learningChart.data.labels = episodes;
                learningChart.data.datasets = datasets;
                learningChart.update('none');
                
                // Exploration chart
                const explorationDatasets = data.agents.map(agent => ({
                    label: agent.name + ' Exploration',
                    data: agent.learning_curve.map(point => point.exploration_rate * 100),
                    borderColor: agent.color,
                    backgroundColor: agent.color + '20',
                    fill: false,
                    tension: 0.4
                }));
                
                explorationChart.data.labels = episodes;
                explorationChart.data.datasets = explorationDatasets;
                explorationChart.update('none');
            }
        }
        
        function updateTrackVisualization(data) {
            const canvas = document.getElementById('trackCanvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (data.track && data.track.length > 0) {
                // Draw track
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 3;
                ctx.beginPath();
                
                const scale = Math.min(canvas.width / 800, canvas.height / 600);
                const offsetX = (canvas.width - 800 * scale) / 2;
                const offsetY = (canvas.height - 600 * scale) / 2;
                
                data.track.forEach((point, index) => {
                    const x = point.x * scale + offsetX;
                    const y = point.y * scale + offsetY;
                    
                    if (index === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                });
                ctx.closePath();
                ctx.stroke();
                
                // Draw agents
                data.agents.forEach(agent => {
                    if (agent.current_state) {
                        const pos = agent.current_state.position;
                        const x = pos[0] * scale + offsetX;
                        const y = pos[1] * scale + offsetY;
                        
                        ctx.fillStyle = agent.color;
                        ctx.beginPath();
                        ctx.arc(x, y, 8, 0, Math.PI * 2);
                        ctx.fill();
                        
                        // Speed indicator
                        const speed = agent.current_state.speed || 0;
                        ctx.fillStyle = 'white';
                        ctx.font = '12px Arial';
                        ctx.fillText(speed.toFixed(0), x + 12, y + 4);
                    }
                });
            }
        }
        
        function initializeCharts() {
            // Learning curve chart
            const learningCtx = document.getElementById('learningChart').getContext('2d');
            learningChart = new Chart(learningCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: true, text: 'Learning Curves (Average Reward)', color: 'white' },
                        legend: { labels: { color: 'white' } }
                    },
                    scales: {
                        x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        y: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
            
            // Exploration chart
            const explorationCtx = document.getElementById('explorationChart').getContext('2d');
            explorationChart = new Chart(explorationCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: true, text: 'Exploration Rate (%)', color: 'white' },
                        legend: { labels: { color: 'white' } }
                    },
                    scales: {
                        x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        y: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' }, min: 0, max: 100 }
                    }
                }
            });
            
            // Real-time chart
            const realtimeCtx = document.getElementById('realtimeChart').getContext('2d');
            realtimeChart = new Chart(realtimeCtx, {
                type: 'line',
                data: { 
                    labels: Array.from({length: 50}, (_, i) => i),
                    datasets: [{
                        label: 'Recent Rewards',
                        data: Array(50).fill(0),
                        borderColor: '#3498db',
                        backgroundColor: '#3498db20',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                        title: { display: true, text: 'Real-time Reward Stream', color: 'white' },
                        legend: { labels: { color: 'white' } }
                    },
                    scales: {
                        x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        y: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
        }
        
        function formatTime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        
        async function startTraining() {
            try {
                const response = await fetch('/api/training/start', { method: 'POST' });
                const data = await response.json();
                console.log('Training started:', data);
            } catch (error) {
                console.error('Error starting training:', error);
            }
        }
        
        async function stopTraining() {
            try {
                const response = await fetch('/api/training/stop', { method: 'POST' });
                const data = await response.json();
                console.log('Training stopped:', data);
            } catch (error) {
                console.error('Error stopping training:', error);
            }
        }
        
        // Initialize
        connectWebSocket();
        
        // Auto-resize track canvas
        window.addEventListener('resize', () => {
            if (dashboardData) {
                updateTrackVisualization(dashboardData);
            }
        });
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    logger.info("Starting TrackMania RL Learning Dashboard...")
    uvicorn.run(app, host="0.0.0.0", port=4000, log_level="info")