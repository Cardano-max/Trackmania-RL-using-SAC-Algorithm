#!/usr/bin/env python3
"""
Simple script to monitor RL training progress
"""

import sys
import time
sys.path.append('tmrl_enhanced')

from learning_dashboard import dashboard

def monitor_training():
    print("🏁 TrackMania RL Training Monitor")
    print("=" * 50)
    
    while True:
        try:
            # Get basic info without complex serialization
            training_active = dashboard.training_active
            current_episode = dashboard.current_episode
            total_time = dashboard.total_training_time
            
            print(f"\r🎯 Episode: {current_episode:3d} | Training: {'🟢 ACTIVE' if training_active else '🔴 STOPPED'} | Time: {total_time:.1f}s", end="", flush=True)
            
            # Show agent performance if available
            if dashboard.active_agents:
                agent_count = len(dashboard.active_agents)
                
                # Get latest rewards
                latest_rewards = []
                for agent_id, agent_data in dashboard.active_agents.items():
                    if agent_data["episode_rewards"]:
                        latest_rewards.append(agent_data["episode_rewards"][-1])
                
                if latest_rewards:
                    avg_reward = sum(latest_rewards) / len(latest_rewards)
                    max_reward = max(latest_rewards)
                    print(f" | Agents: {agent_count} | Avg Reward: {avg_reward:.1f} | Best: {max_reward:.1f}", end="")
            
            if not training_active and current_episode > 0:
                print("\n\n🏆 Training completed!")
                
                # Show final results
                if dashboard.active_agents:
                    print("\n🎯 Final Results:")
                    for agent_id, agent_data in dashboard.active_agents.items():
                        if agent_data["episode_rewards"]:
                            avg_reward = sum(agent_data["episode_rewards"][-10:]) / min(10, len(agent_data["episode_rewards"]))
                            best_reward = max(agent_data["episode_rewards"])
                            episodes = len(agent_data["episode_rewards"])
                            print(f"  🏎️  {agent_data['name']}: {episodes} episodes, Avg: {avg_reward:.1f}, Best: {best_reward:.1f}")
                break
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n\n⏹️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    monitor_training()