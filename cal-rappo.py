"""
CAL-RAPPO: CNN-AdaBN-LSTM and Risk-Aware Proximal Policy Optimization
Weather-robust end-to-end autonomous driving at unsignalized intersections.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random


class AdaBN(nn.Module):
    """
    Adaptive Batch Normalization for weather-invariant feature extraction.
    Maintains separate statistics for each weather condition.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Affine parameters (shared across all weathers)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Statistics for each weather condition
        self.weather_stats = {}
        self.current_weather = 'clear'
    
    def forward(self, x, weather=None):
        """
        Forward pass with adaptive normalization based on weather.
        """
        if weather is None:
            weather = self.current_weather
        
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean([0, 2, 3])
            batch_var = x.var([0, 2, 3], unbiased=False)
            
            # Update weather-specific statistics
            if weather not in self.weather_stats:
                self.weather_stats[weather] = {
                    'running_mean': batch_mean.detach().clone(),
                    'running_var': batch_var.detach().clone(),
                    'count': 1
                }
            else:
                stats = self.weather_stats[weather]
                stats['running_mean'] = (1 - self.momentum) * stats['running_mean'] + self.momentum * batch_mean.detach()
                stats['running_var'] = (1 - self.momentum) * stats['running_var'] + self.momentum * batch_var.detach()
                stats['count'] += 1
            
            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Use stored weather statistics
            if weather in self.weather_stats:
                stats = self.weather_stats[weather]
                mean = stats['running_mean']
                var = stats['running_var']
            else:
                # Fallback to global mean if weather not seen
                if hasattr(self, 'running_mean') and hasattr(self, 'running_var'):
                    mean = self.running_mean
                    var = self.running_var
                else:
                    # Emergency fallback: use batch stats
                    mean = x.mean([0, 2, 3])
                    var = x.var([0, 2, 3], unbiased=False)
        
        # Normalize
        x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return self.weight[None, :, None, None] * x_norm + self.bias[None, :, None, None]
    
    def set_weather(self, weather):
        """Set current weather for inference."""
        self.current_weather = weather
    
    def get_weather_stats(self):
        """Get all weather statistics."""
        return self.weather_stats


class WeatherAwareCNN(nn.Module):
    """
    Weather-aware CNN with AdaBN for spatial feature extraction.
    """
    def __init__(self, img_height=120, img_width=160, input_channels=3):
        super(WeatherAwareCNN, self).__init__()
        
        # Convolutional layers with AdaBN
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.adabn1 = AdaBN(32)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.adabn2 = AdaBN(64)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.adabn3 = AdaBN(128)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Compute output size
        self._compute_output_size(img_height, img_width)
        
        # Fully connected layer for feature projection
        self.fc = nn.Linear(self.flatten_size, 256)
    
    def _compute_output_size(self, h, w):
        """Compute output size after convolutions."""
        # Conv1: kernel=8, stride=4
        h = (h - 8) // 4 + 1
        w = (w - 8) // 4 + 1
        # Pool1: kernel=2, stride=2
        h = h // 2
        w = w // 2
        # Conv2: kernel=4, stride=2
        h = (h - 4) // 2 + 1
        w = (w - 4) // 2 + 1
        # Pool2: kernel=2, stride=2
        h = h // 2
        w = w // 2
        # Conv3: kernel=3, stride=1
        h = h - 3 + 1
        w = w - 3 + 1
        
        self.flatten_size = 128 * h * w
        self.feature_height = h
        self.feature_width = w
    
    def forward(self, x, weather=None):
        """Forward pass through the CNN with weather adaptation."""
        x = self.conv1(x)
        x = self.adabn1(x, weather)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.adabn2(x, weather)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.adabn3(x, weather)
        x = self.relu(x)
        
        # Flatten and project
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def set_weather(self, weather):
        """Set current weather for all AdaBN layers."""
        self.adabn1.set_weather(weather)
        self.adabn2.set_weather(weather)
        self.adabn3.set_weather(weather)
    
    def get_weather_stats(self):
        """Get all weather statistics."""
        return {
            'adabn1': self.adabn1.get_weather_stats(),
            'adabn2': self.adabn2.get_weather_stats(),
            'adabn3': self.adabn3.get_weather_stats()
        }


class WeatherAwarePerception(nn.Module):
    """
    Weather-aware perception module: CNN with AdaBN + LSTM for temporal fusion.
    """
    def __init__(self, img_height=120, img_width=160, input_channels=3,
                 lstm_hidden_size=256, stack_frames=3, ego_state_dim=4):
        super(WeatherAwarePerception, self).__init__()
        
        self.stack_frames = stack_frames
        self.ego_state_dim = ego_state_dim
        self.lstm_hidden_size = lstm_hidden_size
        
        # Spatial feature extractor
        self.cnn = WeatherAwareCNN(img_height, img_width, input_channels)
        
        # Temporal fusion (LSTM)
        self.lstm = nn.LSTM(
            input_size=256 + ego_state_dim,  # CNN features + ego state
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Initialize hidden state
        self.hidden = None
    
    def forward(self, images, ego_state, weather=None):
        """
        Forward pass through the perception module.
        Args:
            images: (batch, frames, channels, H, W) or (frames, channels, H, W)
            ego_state: (batch, frames, ego_dim) or (frames, ego_dim)
            weather: weather condition string
        Returns:
            spatiotemporal feature vector (batch, lstm_hidden_size)
        """
        batch_size = images.shape[0] if len(images.shape) > 4 else 1
        stack_frames = images.shape[1] if len(images.shape) > 4 else images.shape[0]
        
        # Process each frame
        frame_features = []
        for t in range(stack_frames):
            if len(images.shape) > 4:
                frame = images[:, t, :, :, :]
                ego_t = ego_state[:, t, :]
            else:
                frame = images[t, :, :, :].unsqueeze(0)
                ego_t = ego_state[t, :].unsqueeze(0)
            
            # Spatial feature extraction
            cnn_feat = self.cnn(frame, weather)
            
            # Concatenate with ego state
            combined = torch.cat([cnn_feat, ego_t], dim=-1)
            frame_features.append(combined)
        
        # Stack frames
        if len(images.shape) > 4:
            stacked = torch.stack(frame_features, dim=1)  # (batch, frames, feat_dim)
        else:
            stacked = torch.stack(frame_features, dim=0).unsqueeze(0)  # (1, frames, feat_dim)
        
        # LSTM for temporal fusion
        lstm_out, self.hidden = self.lstm(stacked, self.hidden)
        
        # Return last hidden state
        return lstm_out[:, -1, :]
    
    def reset_hidden(self, batch_size=1):
        """Reset LSTM hidden state."""
        self.hidden = None
    
    def set_weather(self, weather):
        """Set current weather for CNN."""
        self.cnn.set_weather(weather)
    
    def get_weather_stats(self):
        """Get all weather statistics from CNN."""
        return self.cnn.get_weather_stats()


class RiskPredictor(nn.Module):
    """
    Risk predictor network that estimates instantaneous risk for state-action pairs.
    """
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(RiskPredictor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, state, action):
        """Predict risk for given state-action pair."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for continuous action space.
    """
    def __init__(self, state_dim, action_dim, hidden_size=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
    
    def forward(self, state):
        """Forward pass to get action distribution."""
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return Normal(mean, std)
    
    def sample(self, state):
        """Sample action from policy."""
        dist = self.forward(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def get_action(self, state, deterministic=False):
        """Get action (stochastic or deterministic)."""
        dist = self.forward(state)
        if deterministic:
            return dist.mean
        return dist.rsample()


class ValueNetwork(nn.Module):
    """
    Value network for state value estimation.
    """
    def __init__(self, state_dim, hidden_size=256):
        super(ValueNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        """Estimate state value."""
        return self.net(state).squeeze(-1)


class CALRAPPOAgent:
    """
    CAL-RAPPO Agent: Weather-aware perception + Risk-constrained PPO.
    """
    def __init__(self, state_dim, action_dim, config):
        super(CALRAPPOAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Extract hyperparameters
        self.lr_actor = config['decision'].get('lr_actor', 3e-4)
        self.lr_critic = config['decision'].get('lr_critic', 3e-4)
        self.lr_risk = config['decision'].get('lr_risk', 1e-3)
        self.lr_dual = config['decision'].get('lr_dual', 5e-3)
        self.gamma = config['decision'].get('gamma', 0.99)
        self.gae_lambda = config['decision'].get('gae_lambda', 0.95)
        self.clip_eps = config['decision'].get('clip_eps', 0.2)
        self.update_epochs = config['decision'].get('update_epochs', 10)
        self.batch_size = config['decision'].get('batch_size', 256)
        self.rollout_steps = config['decision'].get('rollout_steps', 2048)
        self.risk_threshold = config['decision'].get('risk_threshold', 0.2)
        self.entropy_coef = config['decision'].get('entropy_coef', 0.05)
        self.value_coef = config['decision'].get('value_coef', 0.5)
        self.risk_coef = config['decision'].get('risk_coef', 1.0)
        self.max_grad_norm = config['decision'].get('max_grad_norm', 0.5)
        
        # Networks
        self.policy = GaussianPolicy(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.risk_predictor = RiskPredictor(state_dim, action_dim)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_actor)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr_critic)
        self.risk_optimizer = torch.optim.Adam(self.risk_predictor.parameters(), lr=self.lr_risk)
        
        # Lagrange multiplier
        self.dual_lambda = torch.tensor(0.0, requires_grad=False)
        
        # Rollout buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'risks': []
        }
        
        # Perception module (for feature extraction)
        self.perception = WeatherAwarePerception(
            img_height=config['camera']['img_height'],
            img_width=config['camera']['img_width'],
            input_channels=3,
            lstm_hidden_size=config['perception']['lstm_hidden_size'],
            stack_frames=config['perception']['stack_frames'],
            ego_state_dim=4  # speed, x, y, yaw
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_device()
        
        self.total_steps = 0
    
    def to_device(self):
        """Move all networks to device."""
        self.policy.to(self.device)
        self.value.to(self.device)
        self.risk_predictor.to(self.device)
        self.perception.to(self.device)
    
    def extract_features(self, images, ego_state, weather='clear'):
        """
        Extract spatiotemporal features from images and ego state.
        """
        with torch.no_grad():
            # Convert to tensor
            if not isinstance(images, torch.Tensor):
                images = torch.FloatTensor(images)
            if not isinstance(ego_state, torch.Tensor):
                ego_state = torch.FloatTensor(ego_state)
            
            # Ensure correct dimensions
            if len(images.shape) == 4:  # (frames, C, H, W)
                images = images.unsqueeze(0)  # (1, frames, C, H, W)
            if len(ego_state.shape) == 2:  # (frames, dim)
                ego_state = ego_state.unsqueeze(0)  # (1, frames, dim)
            
            images = images.to(self.device)
            ego_state = ego_state.to(self.device)
            
            features = self.perception(images, ego_state, weather)
            return features.cpu().numpy()
    
    def act(self, obs, deterministic=False):
        """
        Select action based on current observation.
        """
        # Extract features
        images = obs['images']
        ego_state = obs['ego_state']
        weather = obs.get('weather', 'clear')
        
        state = self.extract_features(images, ego_state, weather)
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action = self.policy.get_action(state, deterministic=True)
            else:
                action, log_prob = self.policy.sample(state)
        
        action = action.cpu().numpy()
        return action
    
    def store_transition(self, obs, action, reward, next_obs, done, info):
        """Store transition in buffer."""
        # Extract features
        state = self.extract_features(obs['images'], obs['ego_state'], obs.get('weather', 'clear'))
        next_state = self.extract_features(next_obs['images'], next_obs['ego_state'], next_obs.get('weather', 'clear'))
        
        # Compute value and risk
        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.FloatTensor(action).to(self.device)
        
        with torch.no_grad():
            value = self.value(state_t).cpu().numpy()
            risk = self.risk_predictor(state_t, action_t).cpu().numpy()
        
        # Store
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['dones'].append(float(done))
        self.buffer['values'].append(value)
        self.buffer['risks'].append(risk)
        
        self.total_steps += 1
    
    def compute_gae(self, rewards, values, dones, next_values):
        """
        Compute Generalized Advantage Estimation.
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[-1]
                next_done = dones[-1]
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """Update policy and value networks using collected rollouts."""
        if len(self.buffer['states']) < self.rollout_steps:
            return
        
        # Convert buffer to numpy arrays
        states = np.array(self.buffer['states'])
        actions = np.array(self.buffer['actions'])
        rewards = np.array(self.buffer['rewards'])
        next_states = np.array(self.buffer['next_states'])
        dones = np.array(self.buffer['dones'])
        old_values = np.array(self.buffer['values']).flatten()
        old_risks = np.array(self.buffer['risks']).flatten()
        
        # Compute next values
        next_values = []
        with torch.no_grad():
            for ns in next_states:
                ns_t = torch.FloatTensor(ns).to(self.device)
                nv = self.value(ns_t).cpu().numpy()
                next_values.append(nv)
        next_values = np.array(next_values).flatten()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones, next_values)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        old_risks_t = torch.FloatTensor(old_risks).to(self.device)
        
        # Compute old log probs
        with torch.no_grad():
            old_dist = self.policy(states_t)
            old_log_probs = old_dist.log_prob(actions_t).sum(dim=-1)
        
        # PPO update
        for _ in range(self.update_epochs):
            # Shuffle data
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Policy loss
                dist = self.policy(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clip_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(ratio * batch_advantages, clip_ratio * batch_advantages).mean()
                
                # Entropy bonus
                entropy_loss = -self.entropy_coef * entropy
                
                # Risk constraint
                current_risks = self.risk_predictor(batch_states, batch_actions)
                risk_loss = self.dual_lambda * torch.max(
                    current_risks.mean() - self.risk_threshold,
                    torch.tensor(0.0, device=self.device)
                )
                
                total_policy_loss = policy_loss + entropy_loss + risk_loss
                
                # Value loss
                values = self.value(batch_states)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Risk predictor loss
                risk_pred = self.risk_predictor(batch_states, batch_actions)
                risk_target = self.compute_risk_target(batch_states, batch_actions)
                risk_loss_supervised = F.mse_loss(risk_pred, risk_target)
                
                # Update policy
                self.policy_optimizer.zero_grad()
                total_policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # Update risk predictor
                self.risk_optimizer.zero_grad()
                risk_loss_supervised.backward()
                torch.nn.utils.clip_grad_norm_(self.risk_predictor.parameters(), self.max_grad_norm)
                self.risk_optimizer.step()
        
        # Update dual lambda (Lagrange multiplier)
        with torch.no_grad():
            avg_risk = old_risks_t.mean()
            self.dual_lambda = torch.clamp(
                self.dual_lambda + self.lr_dual * (avg_risk - self.risk_threshold),
                min=0.0
            )
        
        # Clear buffer
        self.clear_buffer()
    
    def compute_risk_target(self, state, action):
        """
        Compute risk target for risk predictor training.
        """
        # This is a simplified version; in practice, compute from environment
        # Safety risk, efficiency risk, weather robustness risk
        batch_size = state.shape[0]
        
        # Placeholder: compute based on state and action
        # In real implementation, this should use actual environment information
        risk_target = torch.zeros(batch_size, device=self.device)
        
        # Safety risk: penalize abrupt actions
        # (in practice, need to access previous actions)
        
        # Efficiency risk: penalize low speed
        # (in practice, need to access speed from state)
        
        # Weather robustness risk: feature variance
        # (in practice, need to access weather features)
        
        return risk_target
    
    def clear_buffer(self):
        """Clear rollout buffer."""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'risks': []
        }
    
    def get_checkpoint(self):
        """Get model checkpoint for saving."""
        return {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'risk_predictor_state_dict': self.risk_predictor.state_dict(),
            'perception_state_dict': self.perception.state_dict(),
            'dual_lambda': self.dual_lambda,
            'total_steps': self.total_steps
        }
    
    def load_checkpoint(self, checkpoint):
        """Load model from checkpoint."""
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.risk_predictor.load_state_dict(checkpoint['risk_predictor_state_dict'])
        self.perception.load_state_dict(checkpoint['perception_state_dict'])
        self.dual_lambda = checkpoint.get('dual_lambda', torch.tensor(0.0))
        self.total_steps = checkpoint.get('total_steps', 0)
        self.to_device()
    
    def set_weather(self, weather):
        """Set weather for perception module."""
        self.perception.set_weather(weather)
