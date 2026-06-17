"""
Evaluation script for CAL-RAPPO agent in CARLA unsignalized intersection environment.
"""

import os
import sys
import argparse
import yaml
import csv
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from carla_env import CarlaIntersectionEnv
from cal_rappo import CALRAPPOAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate CAL-RAPPO agent')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results', 
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default='default.yaml', 
                       help='Configuration file path')
    parser.add_argument('--episodes', type=int, default=50, 
                       help='Number of episodes per weather condition')
    parser.add_argument('--host', type=str, default='localhost', 
                       help='CARLA server host')
    parser.add_argument('--port', type=int, default=6000, 
                       help='CARLA server port')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_directories(output_dir):
    """Create output directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'detailed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)


def plot_umap_comparison(features_with_adabn, features_without_adabn, output_dir):
    """Plot UMAP visualization comparison."""
    try:
        import umap
        
        # Create UMAP visualizations
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Without AdaBN
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_without = reducer.fit_transform(features_without_adabn)
        
        # With AdaBN
        embedding_with = reducer.fit_transform(features_with_adabn)
        
        # Plot
        axes[0].scatter(embedding_without[:, 0], embedding_without[:, 1], 
                       c='blue', alpha=0.6, s=5)
        axes[0].set_title('Without AdaBN')
        
        axes[1].scatter(embedding_with[:, 0], embedding_with[:, 1], 
                       c='green', alpha=0.6, s=5)
        axes[1].set_title('With AdaBN')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visualizations', 'umap_comparison.png'), dpi=300)
        plt.close()
        
    except ImportError:
        print('UMAP not available, skipping visualization')


def plot_drivew_boxplot(results, output_dir):
    """Plot driveW_i boxplot comparison."""
    data = []
    labels = []
    
    for algorithm, values in results.items():
        data.append(values['drivew_scores'])
        labels.append(algorithm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, labels=labels)
    ax.set_ylabel('driveW_i (Weather Robustness Score)')
    ax.set_title('Weather Robustness Score Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'drivew_boxplot.png'), dpi=300)
    plt.close()


def evaluate():
    """Main evaluation function."""
    args = parse_args()
    config = load_config(args.config)
    setup_output_directories(args.output)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Initialize environment
    env = CarlaIntersectionEnv(config, args.host, args.port)
    env.setup()
    
    # Initialize agent
    state_dim = config['perception']['lstm_hidden_size']
    action_dim = 2
    agent = CALRAPPOAgent(state_dim, action_dim, config)
    agent.load_checkpoint(checkpoint)
    
    # Evaluation weather conditions
    eval_weathers = config['evaluation']['eval_weathers']
    eval_episodes = args.episodes
    
    # Results storage
    all_results = []
    
    # Detailed results file
    detailed_file = open(os.path.join(args.output, 'detailed', 'episode_details.csv'), 'w', newline='')
    detailed_writer = csv.writer(detailed_file)
    detailed_writer.writerow(['weather', 'episode', 'success', 'collision', 'steps', 
                            'avg_speed', 'completion_time', 'drivew_score', 'reward'])
    
    print(f'Evaluating on {len(eval_weathers)} weather conditions...')
    
    for weather_idx, weather in enumerate(eval_weathers):
        print(f'\n=== Evaluating on {weather} ===')
        
        weather_success = 0
        weather_collision = 0
        weather_rewards = []
        weather_speeds = []
        weather_times = []
        weather_drivew = []
        
        for episode in tqdm(range(eval_episodes), desc=f'{weather}'):
            # Set weather
            env.set_weather(weather)
            
            obs = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            success = False
            collision = False
            
            while not done and steps < env.max_episode_steps:
                action = agent.act(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1
            
            # Record results
            success = info.get('success', False)
            collision = info.get('collision', False)
            
            if success:
                weather_success += 1
                weather_speeds.append(info.get('avg_speed', 0.0))
                weather_times.append(info.get('completion_time', 0.0))
            
            if collision:
                weather_collision += 1
            
            weather_rewards.append(episode_reward)
            weather_drivew.append(info.get('drivew_score', 0.0))
            
            # Write detailed results
            detailed_writer.writerow([
                weather, episode, int(success), int(collision),
                steps, info.get('avg_speed', 0.0),
                info.get('completion_time', 0.0),
                info.get('drivew_score', 0.0),
                episode_reward
            ])
            detailed_file.flush()
        
        # Summary for this weather
        summary = {
            'weather': weather,
            'success_rate': weather_success / eval_episodes * 100,
            'collision_rate': weather_collision / eval_episodes * 100,
            'avg_reward': np.mean(weather_rewards),
            'avg_speed': np.mean(weather_speeds) if weather_speeds else 0,
            'avg_time': np.mean(weather_times) if weather_times else 0,
            'avg_drivew': np.mean(weather_drivew)
        }
        all_results.append(summary)
        
        print(f'Success Rate: {summary["success_rate"]:.2f}%')
        print(f'Collision Rate: {summary["collision_rate"]:.2f}%')
        print(f'Avg Speed: {summary["avg_speed"]:.2f} m/s')
        print(f'Avg Time: {summary["avg_time"]:.2f} s')
        print(f'Avg driveW_i: {summary["avg_drivew"]:.4f}')
    
    detailed_file.close()
    
    # Write summary results
    summary_file = open(os.path.join(args.output, 'summary.csv'), 'w', newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(['weather', 'success_rate', 'collision_rate', 
                           'avg_reward', 'avg_speed', 'avg_time', 'avg_drivew'])
    
    for result in all_results:
        summary_writer.writerow([
            result['weather'],
            f"{result['success_rate']:.2f}",
            f"{result['collision_rate']:.2f}",
            f"{result['avg_reward']:.2f}",
            f"{result['avg_speed']:.2f}",
            f"{result['avg_time']:.2f}",
            f"{result['avg_drivew']:.4f}"
        ])
    
    summary_file.close()
    
    # Overall statistics
    overall_success = np.mean([r['success_rate'] for r in all_results])
    overall_collision = np.mean([r['collision_rate'] for r in all_results])
    
    print(f'\n=== Overall Performance ===')
    print(f'Average Success Rate: {overall_success:.2f}%')
    print(f'Average Collision Rate: {overall_collision:.2f}%')
    
    # Cleanup
    env.cleanup()
    print(f'\nResults saved to {args.output}/')


if __name__ == '__main__':
    evaluate()
