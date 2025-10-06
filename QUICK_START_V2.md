# 🚀 TrackMania RL Two-Container System - Quick Start

## 🎯 **What This Delivers for Henrique:**

✅ **Modular Architecture**: Environment + Model containers communicate via REST API  
✅ **Beautiful Race Replay**: Real-time visualization with multiple agent tracking  
✅ **Professional Demos**: Color-coded agents, smooth playback, speed controls  
✅ **Future-Proof**: Easy environment swapping for new projects  

## 🏗️ **Architecture Overview:**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ENVIRONMENT   │◄──►│      MODEL      │    │     VIEWER      │
│   Port: 8080    │    │   Port: 8081    │    │   Port: 3000    │
│                 │    │                 │    │                 │
│ • TrackMania    │    │ • SAC Agent     │    │ • Race Replay   │
│ • Physics       │    │ • Training      │    │ • Multi-Agent   │
│ • LIDAR         │    │ • Experience    │    │ • Visualization │
│ • Recording     │    │ • Networks      │    │ • Controls      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 **Quick Start:**

### **1. Build and Start:**
```bash
# Build all containers
docker-compose -f docker-compose-v2.yml build

# Start the system
docker-compose -f docker-compose-v2.yml up -d

# Check status
docker-compose -f docker-compose-v2.yml ps
```

### **2. Test the System:**
```bash
python test_two_containers.py
```

### **3. Start Training:**
```bash
# Via API
curl -X POST http://localhost:8081/api/training/start

# Or via web interface at http://localhost:8081
```

### **4. View Race Replays:**
```bash
# Open in browser
open http://localhost:3000
```

## 📡 **Container Communication:**

### **Model → Environment:**
```bash
curl -X POST http://localhost:8080/api/action \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "sac_agent_1",
    "action": {"gas": 0.8, "brake": 0.0, "steering": 0.2},
    "timestamp": "2024-01-01T00:00:00Z"
  }'
```

### **Environment → Model:**
```json
{
  "agent_id": "sac_agent_1",
  "speed": 120.5,
  "position": {"x": 100.0, "y": 50.0, "z": 0.0},
  "lidar": [0.8, 0.9, 0.7, ...],
  "reward": 15.2,
  "done": false,
  "track_completion": 0.25
}
```

## 🎮 **Using the Race Viewer:**

### **Access:** http://localhost:3000

**Features:**
- **Real-time replay** of training sessions
- **Multi-agent tracking** with color coding
- **Smooth playback controls** (0.5x to 4x speed)
- **Agent trails** showing driving patterns
- **Performance metrics** overlay

**Controls:**
- **Play/Pause**: Start/stop replay
- **Speed**: 0.5x, 1x, 2x, 4x playback speeds
- **Scrubbing**: Jump to any frame
- **Race Selection**: Load different recorded sessions

## 🔧 **API Endpoints:**

### **Environment Container (8080):**
- `POST /api/action` - Process agent action
- `POST /api/reset/{agent_id}` - Reset agent
- `POST /api/recording/start` - Start race recording
- `POST /api/recording/stop` - Stop recording
- `GET /api/races` - List available races
- `GET /api/status` - Environment status

### **Model Container (8081):**
- `POST /api/training/start` - Start training
- `POST /api/training/stop` - Stop training
- `GET /api/training/status` - Training metrics
- `POST /api/model/save` - Save model weights
- `POST /api/action/single` - Get single action
- `GET /api/status` - Model status

### **Viewer Container (3000):**
- `GET /api/races` - List races
- `POST /api/load/{race_id}` - Load race
- `POST /api/play` - Start playback
- `POST /api/pause` - Pause playback
- `POST /api/speed/{speed}` - Set speed
- `WebSocket /ws` - Real-time updates

## 📊 **Monitoring:**

### **TensorBoard:** http://localhost:6006
- Training loss curves
- Episode rewards
- Network weights
- Performance metrics

### **Container Logs:**
```bash
# Environment logs
docker logs trackmania-environment -f

# Model logs  
docker logs trackmania-model -f

# Viewer logs
docker logs trackmania-viewer -f
```

## 🎯 **Demonstration Workflow:**

### **1. Start Training Session:**
```bash
# Start recording
curl -X POST http://localhost:8080/api/recording/start

# Start training
curl -X POST http://localhost:8081/api/training/start
```

### **2. Let It Learn (5-10 minutes):**
- Watch training progress in TensorBoard
- Monitor logs for episode completion
- Check training stats via API

### **3. Stop and Replay:**
```bash
# Stop training
curl -X POST http://localhost:8081/api/training/stop

# Stop recording
curl -X POST http://localhost:8080/api/recording/stop

# View replay at http://localhost:3000
```

## 🎨 **Demo Features for Stakeholder Presentations:**

### **Beautiful Visualization:**
- **Color-coded agents** with distinct trails
- **Smooth camera tracking** and movement
- **Professional UI** with controls
- **Real-time metrics** display

### **Interactive Controls:**
- **Playback speed** adjustment
- **Frame-by-frame** stepping
- **Multiple race** comparison
- **Full-screen mode** for presentations

### **Export Capabilities:**
- **Race data** in JSON format
- **Performance metrics** for analysis
- **Screenshots** and video recording
- **Shareable replays** for remote viewing

## 🔄 **Environment Swapping:**

To add new environments (future projects):

1. **Create new environment container** with same API
2. **Update docker-compose** with new service
3. **Model container works unchanged** 
4. **Viewer shows any environment** automatically

Example for a new racing game:
```yaml
new-racing-env:
  build: ./new-racing-environment
  ports: ["8082:8080"]  # Same API, different port
```

## 🎉 **Success Metrics:**

✅ **Containers communicate** seamlessly via REST API  
✅ **Training loop** runs automatically with environment  
✅ **Race recording** captures agent behavior  
✅ **Beautiful replay** with multiple agents and trails  
✅ **Professional interface** ready for stakeholder demos  
✅ **Modular design** allows easy environment swapping  

## 🚀 **Ready for Production:**

- **Health checks** ensure reliable operation
- **Auto-restart** policies for robustness  
- **Shared volumes** for data persistence
- **Network isolation** for security
- **Scalable architecture** for multiple agents
- **Monitoring integration** with TensorBoard

This system delivers exactly what Henrique requested: beautiful demonstrations of AI learning with professional visualization capabilities! 🎯