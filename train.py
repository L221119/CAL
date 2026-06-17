"""
Training script for CAL-RAPPO agent in CARLA unsignalized intersection environment.
"""

import os
import sys
import argparse
import yaml
import csv
import time
import subprocess
import signal
import psutil
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from carla_env import CarlaIntersectionEnv
from cal_rappo import CALRAPPOAgent
from traffic_manager import TrafficManager


# ============================================================
# CARLA Server Auto-Start Functions
# ============================================================

def get_carla_root():
    """Get CARLA_ROOT from environment variable."""
    carla_root = os.environ.get('CARLA_ROOT')
    if carla_root is None:
        print('WARNING: CARLA_ROOT environment variable not set!')
        print('Please set it before running:')
        print('  Windows: $env:CARLA_ROOT="D:\\CARLA_0.9.12"')
        print('  Linux: export CARLA_ROOT="/path/to/CARLA_0.9.12"')
        return None
    return carla_root


def find_carla_server(carla_root):
    """Find CARLA server executable."""
    # Common CARLA server executable names
    possible_names = [
        'CarlaUE4.exe',  # Windows
        'CarlaUE4.sh',   # Linux
        'CarlaUE4',      # Linux (without extension)
    ]
    
    for name in possible_names:
        full_path = os.path.join(carla_root, name)
        if os.path.exists(full_path):
            return full_path
    
    # Try alternative path
    for name in possible_names:
        full_path = os.path.join(carla_root, 'CarlaUE4', name)
        if os.path.exists(full_path):
            return full_path
    
    return None


def is_carla_running(port=6000):
    """Check if CARLA server is already running on specified port."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0


def start_carla_server(carla_root, port=6000, timeout=30):
    """
    Start CARLA server automatically.
    
    Args:
        carla_root: Path to CARLA installation
        port: Port to run CARLA on
        timeout: Maximum time to wait for server to start
    
    Returns:
        process: Popen object for CARLA process, or None if failed
    """
    # Check if already running
    if is_carla_running(port):
        print(f'CARLA server already running on port {port}')
        return None
    
    # Find CARLA server executable
    server_path = find_carla_server(carla_root)
    if server_path is None:
        print(f'ERROR: Could not find CARLA server executable in {carla_root}')
        print('Please ensure CARLA is installed correctly.')
        return None
    
    print(f'Starting CARLA server from: {server_path}')
    print(f'  Port: {port}')
    print(f'  Timeout: {timeout}s')
    
    try:
        # Start CARLA server
        # Use -carla-rpc-port to specify port
        process = subprocess.Popen(
            [server_path, f'-carla-rpc-port={port}', '-quality-level=Low'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=carla_root
        )
        
        # Wait for server to start
        print('Waiting for CARLA server to start...')
        start_time = time.time()
        while time.time() - start_time < timeout:
            if is_carla_running(port):
                print(f'CARLA server started successfully on port {port}')
                return process
            time.sleep(0.5)
        
        print(f'WARNING: CARLA server did not respond within {timeout}s')
        print('Check if server is running manually.')
        return process
        
    except Exception as e:
        print(f'ERROR: Failed to start CARLA server: {e}')
        return None


def stop_carla_server(process):
    """Stop CARLA server process."""
    if process is None:
        return
    
    print('Stopping CARLA server...')
    try:
        # Try graceful shutdown first
        process.terminate()
        process.wait(timeout=5)
        print('CARLA server stopped.')
    except subprocess.TimeoutExpired:
        # Force kill if not responding
        process.kill()
        print('CARLA server force killed.')
    except Exception as e:
        print(f'Error stopping CARLA server: {e}')


# ============================================================
# Main Training Code
# ============================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CAL-RAPPO agent')
    
    # CARLA server
    parser.add_argument('--no_auto_start', action='store_true', 
                       help='Disable automatic CARLA server start')
    parser.add_argument('--carla_timeout', type=int, default=30,
                       help='Timeout for CARLA server startup (seconds)')
    
    # Environment
    parser.add_argument('--host', type=str, default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=6000, help='CARLA server port')
    parser.add_argument('--town', type=str, default='Town05', help='CARLA town map')
    parser.add_argument('--fps', type=int, default=10, help='Simulation FPS')
    
    # ... 其他参数保持不变 ...
    # Camera
    parser.add_argument('--img_width', type=int, default=256, help='Camera image width')
    parser.add_argument('--img_height', type=int, default=256, help='Camera image height')
    
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

    parser.add_argument('--task', type=str, default='straight', 
                       choices=['straight', 'left', 'right', 'uturn'], 
                       help='Task to train on')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(save_path, tensorboard_log, task):
    """Create necessary directories."""
    task_save_path = os.path.join(save_path, task)
    task_log_path = os.path.join(tensorboard_log, task)
    os.makedirs(task_save_path, exist_ok=True)
    os.makedirs(task_log_path, exist_ok=True)
    os.makedirs('date', exist_ok=True)
    return task_save_path, task_log_path


def save_checkpoint(agent, episode, save_path, task, is_final=False):
    """Save model checkpoint."""
    checkpoint = agent.get_checkpoint()
    checkpoint['task'] = task
    
    if is_final:
        filename = os.path.join(save_path, f'final_model.pth')
    else:
        filename = os.path.join(save_path, f'checkpoint_{episode}.pth')
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved to {filename}')


def evaluate_agent(agent, env, eval_episodes=10):
    """Evaluate the current agent with comprehensive metrics."""
    success_count = 0
    collision_count = 0
    total_reward = 0.0
    total_steps = 0
    total_speed = 0.0
    total_completion_time = 0.0
    total_drivew = 0.0
    success_episodes = 0
    
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        episode_speed = 0.0
        drivew_score = 0.0
        
        while not done and steps < env.max_episode_steps:
            action = agent.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            episode_speed += info.get('speed', 0.0)
            drivew_score = info.get('drivew_score', 0.0)
        
        total_reward += episode_reward
        total_steps += steps
        total_drivew += drivew_score
        
        if info.get('success', False):
            success_count += 1
            total_speed += episode_speed / max(steps, 1)
            total_completion_time += info.get('completion_time', steps / env.fps)
            success_episodes += 1
        
        if info.get('collision', False):
            collision_count += 1
    
    # Calculate metrics
    success_rate = success_count / eval_episodes * 100
    collision_rate = collision_count / eval_episodes * 100
    
    avg_speed = total_speed / max(success_episodes, 1)
    avg_completion_time = total_completion_time / max(success_episodes, 1)
    avg_drivew = total_drivew / eval_episodes
    
    return {
        'avg_reward': total_reward / eval_episodes,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_speed': avg_speed,
        'completion_time': avg_completion_time,
        'drivew_score': avg_drivew,
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
    
    # Setup directories (with task subdirectory)
    save_path, log_dir = setup_directories(args.save_path, args.tensorboard_log, args.task)
    
    # ============================================================
    # Auto-start CARLA server
    # ============================================================
    carla_process = None
    if not args.no_auto_start:
        carla_root = get_carla_root()
        if carla_root is not None:
            carla_process = start_carla_server(carla_root, args.port, args.carla_timeout)
            if carla_process is None:
                print('WARNING: Could not auto-start CARLA server.')
                print('Please start CARLA manually before training.')
        else:
            print('WARNING: CARLA_ROOT not set. Auto-start disabled.')
            print('Please start CARLA manually or set CARLA_ROOT environment variable.')
    
    try:
        # Initialize TensorBoard writer
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(os.path.join(log_dir, run_id))
        
        # Initialize CSV logger
        csv_path = os.path.join('date', f'train_{args.task}_{run_id}.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['episode', 'task', 'total_reward', 'avg_reward', 
                            'success', 'collision', 'avg_speed', 'completion_time', 
                            'drivew_score', 'steps'])
        
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
        print(f'\n{"="*60}')
        print(f'Starting training for task: {args.task}')
        print(f'Total episodes: {args.total_episodes}')
        print(f'{"="*60}\n')
        
        best_success_rate = 0.0
        
        for episode in range(1, args.total_episodes + 1):
            # Reset environment with random weather
            obs = env.reset()
            agent.reset_hidden()
            
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
                args.task,
                episode_reward,
                episode_reward / max(episode_steps, 1),
                int(info.get('success', False)),
                int(info.get('collision', False)),
                info.get('avg_speed', 0.0),
                info.get('completion_time', 0.0),
                info.get('drivew_score', 0.0),
                episode_steps
            ])
            csv_file.flush()
            
            # TensorBoard logging
            writer.add_scalar(f'{args.task}/Episode/Total_Reward', episode_reward, episode)
            writer.add_scalar(f'{args.task}/Episode/Steps', episode_steps, episode)
            writer.add_scalar(f'{args.task}/Episode/Success', float(info.get('success', False)), episode)
            writer.add_scalar(f'{args.task}/Episode/Collision', float(info.get('collision', False)), episode)
            writer.add_scalar(f'{args.task}/Episode/Avg_Speed', info.get('avg_speed', 0.0), episode)
            writer.add_scalar(f'{args.task}/Episode/DriveW_Score', info.get('drivew_score', 0.0), episode)
            
            # Print progress
            if episode % 10 == 0:
                print(f'[{args.task}] Episode {episode}/{args.total_episodes}, '
                      f'Reward: {episode_reward:.2f}, '
                      f'Steps: {episode_steps}, '
                      f'Success: {info.get("success", False)}, '
                      f'Collision: {info.get("collision", False)}')
            
            # Periodic evaluation
            if episode % args.eval_freq == 0:
                eval_results = evaluate_agent(agent, env, eval_episodes=20)
                
                # TensorBoard logging
                writer.add_scalar(f'{args.task}/Evaluation/Avg_Reward', eval_results['avg_reward'], episode)
                writer.add_scalar(f'{args.task}/Evaluation/Success_Rate', eval_results['success_rate'], episode)
                writer.add_scalar(f'{args.task}/Evaluation/Collision_Rate', eval_results['collision_rate'], episode)
                writer.add_scalar(f'{args.task}/Evaluation/Avg_Speed', eval_results['avg_speed'], episode)
                writer.add_scalar(f'{args.task}/Evaluation/Completion_Time', eval_results['completion_time'], episode)
                writer.add_scalar(f'{args.task}/Evaluation/DriveW_Score', eval_results['drivew_score'], episode)
                
                print(f'\n{"="*50}')
                print(f'[{args.task}] Evaluation at Episode {episode}')
                print(f'{"="*50}')
                print(f'Avg Reward: {eval_results["avg_reward"]:.2f}')
                print(f'Success Rate: {eval_results["success_rate"]:.2f}%')
                print(f'Collision Rate: {eval_results["collision_rate"]:.2f}%')
                print(f'Avg Speed: {eval_results["avg_speed"]:.2f} m/s')
                print(f'Completion Time: {eval_results["completion_time"]:.2f} s')
                print(f'DriveW Score: {eval_results["drivew_score"]:.4f}')
                print(f'Avg Steps: {eval_results["avg_steps"]:.1f}')
                print(f'{"="*50}\n')
                
                # Save best model
                if eval_results['success_rate'] > best_success_rate:
                    best_success_rate = eval_results['success_rate']
                    save_checkpoint(agent, episode, save_path, args.task, is_final=False)
            
            # Periodic checkpoint
            if episode % args.checkpoint_freq == 0:
                save_checkpoint(agent, episode, save_path, args.task, is_final=False)
        
        # Save final model
        save_checkpoint(agent, args.total_episodes, save_path, args.task, is_final=True)
        
        # Print final summary
        print(f'\n{"="*60}')
        print(f'Training completed for task: {args.task}')
        print(f'Best Success Rate: {best_success_rate:.2f}%')
        print(f'{"="*60}\n')
        
        # Cleanup
        csv_file.close()
        env.cleanup()
        writer.close()
        
    except KeyboardInterrupt:
        print('\nTraining interrupted by user. Saving checkpoint...')
        if 'agent' in locals():
            save_checkpoint(agent, episode, save_path, args.task, is_final=False)
    
    finally:
        # Stop CARLA server
        if carla_process is not None:
            stop_carla_server(carla_process)


if __name__ == '__main__':
    train()
