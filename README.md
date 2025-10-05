# TrackMania RL using SAC Algorithm

A complete implementation of Soft Actor-Critic (SAC) for autonomous driving in TrackMania. Built from scratch with production-ready Docker infrastructure and comprehensive testing.

**Author**: Muhammad Ateeb Taseer  
**Implementation**: From mathematical foundations without AI assistance

This repository provides a complete, modular Docker setup for training RL agents in TrackMania using the TMRL framework. It includes both the official TMRL implementation and a standalone SAC trainer for testing without the actual game.

## Features

- 🐳 **Fully Dockerized**: Complete setup with server, trainer, and rollout components
- 🧠 **Multiple Training Options**: Official TMRL or standalone SAC implementation
- 📊 **Monitoring**: TensorBoard integration for training visualization
- 🚗 **Mock Environment**: Test training without TrackMania game
- 🔧 **Modular**: Easy to configure and extend

## Prerequisites

- Docker and Docker Compose
- For GPU training: NVIDIA Docker runtime (optional)
- For actual game training: TrackMania 2020 with OpenPlanet

## Quick Start

### 1. Build and Start Services

```bash
cd docker
docker compose build
docker compose up -d
```

This starts:
- **tmrl-server**: Coordination server (port 5555)
- **tmrl-trainer**: Training service (port 5556) 
- **tensorboard**: Monitoring dashboard (port 6006)

### 2. Monitor Training

- **TensorBoard**: http://localhost:6006
- **Logs**: `docker logs tmrl-trainer -f`

### 3. Test Environment (Optional)

To test with mock rollout worker:

```bash
docker compose --profile with-rollout up tmrl-rollout
```

### 4. Connect Real TrackMania Client

On a Windows PC with TrackMania 2020:

```powershell
powershell -ExecutionPolicy Bypass -File .\clients\windows_rollout_client.ps1 -TmrlDataPath "$env:USERPROFILE\TmrlData" -ServerIp "<your-server-ip>" -ServerPort 5555
```

## Configuration

The system uses `/TmrlData/config/config.json` for configuration:

### Basic Security
```json
{
  "PASSWORD": "your-strong-password-here",
  "TLS": false
}
```

### Training Algorithm (SAC)
```json
{
  "ALG": {
    "ALGORITHM": "SAC",
    "LR_ACTOR": 0.0003,
    "LR_CRITIC": 0.0005,
    "GAMMA": 0.995,
    "ALPHA": 0.2
  }
}
```

### Environment Settings
The config includes LIDAR environment setup:
- 19-beam LIDAR observations
- Speed and action history
- Customizable reward function

## Training Modes

### 1. Full TMRL Training
Uses the official TMRL framework with real TrackMania game.

### 2. Standalone SAC Training  
Custom SAC implementation with mock environment for testing:
- No game required
- Immediate feedback
- Algorithm validation

## File Structure

```
trackmania-RL/
├── docker/
│   ├── docker-compose.yml       # Main orchestration
│   ├── server.Dockerfile        # TMRL server
│   ├── trainer.Dockerfile       # Training service
│   ├── rollout.Dockerfile       # Rollout worker
│   └── entrypoints/             # Service entry points
├── scripts/
│   ├── setup_env.py            # Mock environment
│   ├── train_sac.py             # Standalone SAC trainer
│   └── test_environment.py      # Environment tests
├── tmrl_templates/              # Configuration templates
└── clients/                     # Client connection scripts
```

## GPU Support

To enable GPU training, modify the Dockerfiles:

```dockerfile
# Replace CPU PyTorch with GPU version
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

And add GPU access to docker-compose.yml:

```yaml
services:
  tmrl-trainer:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Security Notes

⚠️ **Important**: Default setup uses unencrypted communication for localhost testing.

For production/public networks:
1. Set `"TLS": true` in config.json
2. Generate TLS certificates
3. Use strong passwords
4. Follow TMRL security guidelines

## Monitoring & Debugging

- **Training Progress**: TensorBoard at http://localhost:6006
- **Container Logs**: `docker logs <container-name> -f`
- **Data Volume**: `docker volume inspect trackmania-rl_tmrl_data`
- **Test Environment**: `docker exec -it tmrl-trainer python /app/scripts/test_environment.py`

## Troubleshooting

### Container Issues
```bash
# Check container status
docker compose ps

# View logs
docker logs tmrl-trainer -f

# Rebuild if needed
docker compose build --no-cache
```

### Training Issues
```bash
# Test mock environment
docker exec -it tmrl-trainer python /app/scripts/test_environment.py

# Check configuration
docker exec -it tmrl-trainer cat /TmrlData/config/config.json
```

## Advanced Usage

### Custom Environments
Modify `scripts/setup_env.py` to create custom training environments.

### Algorithm Tuning
Edit hyperparameters in `tmrl_templates/config.trainer.json`.

### Multi-Agent Training
Scale rollout workers by running multiple client connections.

## Contributing

This setup provides a foundation for TrackMania RL research. Contributions welcome for:
- Additional algorithms (PPO, DDPG, etc.)
- Enhanced environments
- Performance optimizations
- Documentation improvements

