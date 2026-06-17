"""
Training script for CAL-RAPPO agent in CARLA unsignalized intersection environment.
"""

import os
import sys
import argparse
import yaml
import csv
import time
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from carla_env import CarlaIntersectionEnv
from cal_rappo import CALRAPPOAgent
from traffic_manager import TrafficManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CAL-RAPPO agent')
    
    # Environment
    parser.add_argument('--host', type=str, default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=6000, help='CARLA server port')
    parser.add_argument('--town', type=str, default='Town05', help='CARLA town map')
    parser.add_argument('--fps', type=int, default=10, help='Simulation FPS')
    
    # Camera
    parser.add_argument('--img_width', type=int, default=160, help='Camera image width')
    parser.add_argument('--img_height', type=int, default=120, help='Camera image height')
    
    # Perception
    parser.add_argument('--stack_frames', type=int, default=3, help='Number of frames to stack')
    parser.add_argument('--lstm_hidden_size', type=int, default=256, help='LSTM hidden size')
    
    # Decision (RAPPO)
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for actor-critic')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clip epsilon')
    
    # Reward weights
    parser.add_argument('--omige_r1', type=float, default=0.5, help='Safety reward weight')
    parser.add_argument('--omige_r2', type=float, default=0.3, help='Efficiency reward weight')
    parser.add_argument('--omige_r3', type=float, default=0.2, help='Weather robustness reward weight')
    
    # Safety
    parser.add_argument('--safe_distance', type=float, default=5.0, help='Safe distance threshold')
    parser.add_argument('--lane_threshold', type=float, default=2.0, help='Lane deviation threshold')
    
    # Training
    parser.add_argument('--total_episodes', type=int, default=500, help='Total training episodes')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--update_epochs', type=int, default=10, help='PPO update epochs')
    parser.add_argument('--rollout_steps', type=int, default=2048, help='Rollout steps per update')
    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint save frequency')
    parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency')
    
    # Agent selection
    parser.add_argument('--agent', type=str, default='cal-rappo', 
                       choices=['cal-rappo', 'ppo', 'td3', 'ddpg', 'd3qn'],
                       help='Agent algorithm to use')
    
    # Paths
    parser.add_argument('--save_path', type=str, default='model', help='Model save directory')
    parser.add_argument('--tensorboard_log', type=str, default='runs', help='TensorBoard log directory')
    parser.add_argument('--config', type=str, default='default.yaml', help='Configuration file path')

    parser.add_argument('--task', type=str, default='straight', choices=['straight', 'left', 'right', 'uturn'], help='Task to train on')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(save_path, tensorboard_log):
    """Create necessary directories."""
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs('date', exist_ok=True)


def save_checkpoint(agent, episode, save_path, is_final=False):
    """Save model checkpoint."""
    checkpoint = agent.get_checkpoint()
    if is_final:
        filename = os.path.join(save_path, 'final_model.pth')
    else:
        filename = os.path.join(save_path, f'checkpoint_{episode}.pth')
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved to {filename}')


def evaluate_agent(agent, env, eval_episodes=10):
    """Evaluate the current agent."""
    success_count = 0
    collision_count = 0
    total_reward = 0.0
    total_steps = 0
    
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done and steps < env.max_episode_steps:
            action = agent.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_reward += episode_reward
        total_steps += steps
        
        if info.get('success', False):
            success_count += 1
        if info.get('collision', False):
            collision_count += 1
    
    avg_reward = total_reward / eval_episodes
    success_rate = success_count / eval_episodes * 100
    collision_rate = collision_count / eval_episodes * 100
    
    return {
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_steps': total_steps / eval_episodes
    }


def train():
    """Main training loop."""
    args = parse_args()
    config = load_config(args.config)
    
    # Update config with command line arguments
    config['camera']['img_width'] = args.img_width
    config['camera']['img_height'] = args.img_height
    config['perception']['stack_frames'] = args.stack_frames
    config['perception']['lstm_hidden_size'] = args.lstm_hidden_size
    config['training']['total_episodes'] = args.total_episodes
    config['training']['checkpoint_freq'] = args.checkpoint_freq
    
    # Setup directories
    setup_directories(args.save_path, args.tensorboard_log)
    
    # Initialize TensorBoard writer
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.tensorboard_log, run_id)
    writer = SummaryWriter(log_dir)
    
    # Initialize CSV logger
    csv_path = os.path.join('date', f'train_{run_id}.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'total_reward', 'avg_reward', 'success_rate', 
                        'collision_rate', 'avg_speed', 'completion_time'])
    
    # Initialize environment
    env = CarlaIntersectionEnv(config, args.host, args.port, task=args.task)
    env.setup()
    
    # Initialize agent
    state_dim = config['perception']['lstm_hidden_size']
    action_dim = 2  # target_speed, steering_angle
    
    if args.agent == 'cal-rappo':
        agent = CALRAPPOAgent(state_dim, action_dim, config)
    else:
        raise ValueError(f'Agent {args.agent} not implemented yet')
    
    # Training loop
    print(f'Starting training for {args.total_episodes} episodes...')
    best_success_rate = 0.0
    
    for episode in range(1, args.total_episodes + 1):
        # Reset environment with random weather
        obs = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        # Collect rollout
        while not done and episode_steps < env.max_episode_steps:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            
            agent.store_transition(obs, action, reward, next_obs, done, info)
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            # Update agent if enough steps collected
            if len(agent.buffer) >= args.rollout_steps:
                agent.update()
        
        # Log episode statistics
        csv_writer.writerow([
            episode,
            episode_reward,
            episode_reward / max(episode_steps, 1),
            info.get('success', False),
            info.get('collision', False),
            info.get('avg_speed', 0.0),
            info.get('completion_time', 0.0)
        ])
        csv_file.flush()
        
        # TensorBoard logging
        writer.add_scalar('Episode/Total_Reward', episode_reward, episode)
        writer.add_scalar('Episode/Steps', episode_steps, episode)
        writer.add_scalar('Episode/Success', float(info.get('success', False)), episode)
        writer.add_scalar('Episode/Collision', float(info.get('collision', False)), episode)
        writer.add_scalar('Episode/Avg_Speed', info.get('avg_speed', 0.0), episode)
        
        # Print progress
        if episode % 10 == 0:
            print(f'Episode {episode}/{args.total_episodes}, '
                  f'Reward: {episode_reward:.2f}, '
                  f'Steps: {episode_steps}, '
                  f'Success: {info.get("success", False)}, '
                  f'Collision: {info.get("collision", False)}')
        
        # Periodic evaluation
        if episode % args.eval_freq == 0:
            eval_results = evaluate_agent(agent, env, eval_episodes=20)
            writer.add_scalar('Evaluation/Avg_Reward', eval_results['avg_reward'], episode)
            writer.add_scalar('Evaluation/Success_Rate', eval_results['success_rate'], episode)
            writer.add_scalar('Evaluation/Collision_Rate', eval_results['collision_rate'], episode)
            
            print(f'\n=== Evaluation at Episode {episode} ===')
            print(f'Avg Reward: {eval_results["avg_reward"]:.2f}')
            print(f'Success Rate: {eval_results["success_rate"]:.2f}%')
            print(f'Collision Rate: {eval_results["collision_rate"]:.2f}%\n')
            
            # Save best model
            if eval_results['success_rate'] > best_success_rate:
                best_success_rate = eval_results['success_rate']
                save_checkpoint(agent, episode, args.save_path, is_final=False)
        
        # Periodic checkpoint
        if episode % args.checkpoint_freq == 0:
            save_checkpoint(agent, episode, args.save_path, is_final=False)
    
    # Save final model
    save_checkpoint(agent, args.total_episodes, args.save_path, is_final=True)
    
    # Cleanup
    csv_file.close()
    env.cleanup()
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    train()
