"""
CAL-RAPPO: CNN-AdaBN-LSTM and Risk-Aware Proximal Policy Optimization
Weather-robust end-to-end autonomous driving at unsignalized intersections.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class AdaBN(nn.Module):
    """Adaptive Batch Normalization for weather-invariant feature extraction."""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.weather_stats = {}
        self.current_weather = 'clear'
        self.running_mean = None
        self.running_var = None
        
        self.weather_map = {
            'glare': 'clear',
            'heavy_rain': 'light_rain',
            'heavy_snow': 'light_snow',
            'haze': 'light_fog'
        }
    
    def forward(self, x, weather=None):
        if weather is None:
            weather = self.current_weather
        
        if not self.training and weather not in self.weather_stats:
            if weather in self.weather_map:
                mapped = self.weather_map[weather]
                if mapped in self.weather_stats:
                    weather = mapped
        
        if self.training:
            batch_mean = x.mean([0, 2, 3])
            batch_var = x.var([0, 2, 3], unbiased=False)
            
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
            
            if self.running_mean is None:
                self.running_mean = batch_mean.detach().clone()
                self.running_var = batch_var.detach().clone()
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
            
            mean, var = batch_mean, batch_var
        else:
            if weather in self.weather_stats:
                stats = self.weather_stats[weather]
                mean, var = stats['running_mean'], stats['running_var']
            elif self.running_mean is not None:
                mean, var = self.running_mean, self.running_var
            else:
                mean, var = x.mean([0, 2, 3]), x.var([0, 2, 3], unbiased=False)
        
        x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return self.weight[None, :, None, None] * x_norm + self.bias[None, :, None, None]
    
    def set_weather(self, weather):
        self.current_weather = weather
    
    def get_weather_stats(self):
        return self.weather_stats


class WeatherAwareCNN(nn.Module):
    """Weather-aware CNN with AdaBN for spatial feature extraction."""
    
    def __init__(self, img_height=256, img_width=256, input_channels=3):
        super(WeatherAwareCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.adabn1 = AdaBN(32)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.adabn2 = AdaBN(64)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.adabn3 = AdaBN(128)
        
        self.relu = nn.ReLU(inplace=True)
        
        self._compute_output_size(img_height, img_width)
        self.fc = nn.Linear(self.flatten_size, 256)
    
    def _compute_output_size(self, h, w):
        h = (h - 8) // 4 + 1
        w = (w - 8) // 4 + 1
        h, w = h // 2, w // 2
        h = (h - 4) // 2 + 1
        w = (w - 4) // 2 + 1
        h, w = h // 2, w // 2
        h, w = h - 3 + 1, w - 3 + 1
        self.flatten_size = 128 * h * w
    
    def forward(self, x, weather=None):
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
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def set_weather(self, weather):
        self.adabn1.set_weather(weather)
        self.adabn2.set_weather(weather)
        self.adabn3.set_weather(weather)
    
    def get_weather_stats(self):
        return {
            'adabn1': self.adabn1.get_weather_stats(),
            'adabn2': self.adabn2.get_weather_stats(),
            'adabn3': self.adabn3.get_weather_stats()
        }


class WeatherAwarePerception(nn.Module):
    """Weather-aware perception module: CNN + LSTM for temporal fusion."""
    
    def __init__(self, img_height=256, img_width=256, input_channels=3,
                 lstm_hidden_size=256, stack_frames=3, ego_state_dim=4):
        super(WeatherAwarePerception, self).__init__()
        
        self.stack_frames = stack_frames
        self.ego_state_dim = ego_state_dim
        self.lstm_hidden_size = lstm_hidden_size
        
        self.cnn = WeatherAwareCNN(img_height, img_width, input_channels)
        
        self.lstm = nn.LSTM(
            input_size=256 + ego_state_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.hidden = None
    
    def forward(self, images, ego_state, weather=None):
        batch_size = images.shape[0] if len(images.shape) > 4 else 1
        stack_frames = images.shape[1] if len(images.shape) > 4 else images.shape[0]
        
        frame_features = []
        
        for t in range(stack_frames):
            if len(images.shape) > 4:
                frame = images[:, t, :, :, :]
                ego_t = ego_state[:, t, :]
            else:
                frame = images[t, :, :, :].unsqueeze(0)
                ego_t = ego_state[t, :].unsqueeze(0)
            
            cnn_feat = self.cnn(frame, weather)
            combined = torch.cat([cnn_feat, ego_t], dim=-1)
            frame_features.append(combined)
        
        if len(images.shape) > 4:
            stacked = torch.stack(frame_features, dim=1)
        else:
            stacked = torch.stack(frame_features, dim=0).unsqueeze(0)
        
        lstm_out, self.hidden = self.lstm(stacked, self.hidden)
        
        return lstm_out[:, -1, :]
    
    def reset_hidden(self):
        self.hidden = None
    
    def set_weather(self, weather):
        self.cnn.set_weather(weather)
    
    def get_weather_stats(self):
        return self.cnn.get_weather_stats()


class RiskPredictor(nn.Module):
    """Risk predictor network for state-action pairs."""
    
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(RiskPredictor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class GaussianPolicy(nn.Module):
    """Gaussian policy network for continuous action space."""
    
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
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return Normal(mean, std)
    
    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def get_action(self, state, deterministic=False):
        dist = self.forward(state)
        return dist.mean if deterministic else dist.rsample()


class ValueNetwork(nn.Module):
    """Value network for state value estimation."""
    
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
        return self.net(state).squeeze(-1)


class CALRAPPOAgent:
    """
    CAL-RAPPO Agent: Weather-aware perception + Risk-constrained PPO.
    """
    
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        dc = config.get('decision', {})
        self.lr_actor = dc.get('lr_actor', 3e-4)
        self.lr_critic = dc.get('lr_critic', 3e-4)
        self.lr_risk = dc.get('lr_risk', 1e-3)
        self.lr_dual = dc.get('lr_dual', 5e-3)
        self.gamma = dc.get('gamma', 0.99)
        self.gae_lambda = dc.get('gae_lambda', 0.95)
        self.clip_eps = dc.get('clip_eps', 0.2)
        self.update_epochs = dc.get('update_epochs', 10)
        self.batch_size = dc.get('batch_size', 256)
        self.rollout_steps = dc.get('rollout_steps', 2048)
        self.risk_threshold = dc.get('risk_threshold', 0.2)
        self.entropy_coef = dc.get('entropy_coef', 0.05)
        self.max_grad_norm = dc.get('max_grad_norm', 0.5)
        
        rc = config.get('reward', {})
        self.risk_safety_weight = rc.get('risk_safety_weight', 0.5)
        self.risk_efficiency_weight = rc.get('risk_efficiency_weight', 0.3)
        self.risk_weather_weight = rc.get('risk_weather_weight', 0.2)
        self.ref_speed = rc.get('ref_speed', 8.0)
        self.weather_variance_threshold = rc.get('weather_variance_threshold', 10.0)
        self.max_action_change = 0.3
        
        self.policy = GaussianPolicy(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.risk_predictor = RiskPredictor(state_dim, action_dim)
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_actor)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr_critic)
        self.risk_optimizer = torch.optim.Adam(self.risk_predictor.parameters(), lr=self.lr_risk)
        
        self.dual_lambda = torch.tensor(0.0, requires_grad=False)
        
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'values': [],
            'risks': [],
            'collisions': [],
            'speeds': [],
            'prev_actions': [],
            'weather_variances': []
        }
        
        cc = config.get('camera', {})
        pc = config.get('perception', {})
        self.perception = WeatherAwarePerception(
            img_height=cc.get('img_height', 256),
            img_width=cc.get('img_width', 256),
            input_channels=3,
            lstm_hidden_size=pc.get('lstm_hidden_size', 256),
            stack_frames=pc.get('stack_frames', 3),
            ego_state_dim=4
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_device()
        self.total_steps = 0
    
    def to_device(self):
        self.policy.to(self.device)
        self.value.to(self.device)
        self.risk_predictor.to(self.device)
        self.perception.to(self.device)
    
    def reset_hidden(self):
        self.perception.reset_hidden()
    
    def extract_features(self, images, ego_state, weather='clear'):
        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                images = torch.FloatTensor(images)
            if not isinstance(ego_state, torch.Tensor):
                ego_state = torch.FloatTensor(ego_state)
            
            if len(images.shape) == 4:
                images = images.unsqueeze(0)
            if len(ego_state.shape) == 2:
                ego_state = ego_state.unsqueeze(0)
            
            images = images.to(self.device)
            ego_state = ego_state.to(self.device)
            
            features = self.perception(images, ego_state, weather)
            return features.cpu().numpy()

    def act(self, obs, deterministic=False):
        images = obs['images']
        ego_state = obs['ego_state']
        weather = obs.get('weather', 'clear')

        state = self.extract_features(images, ego_state, weather)
        state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            if deterministic:
                action = self.policy.get_action(state, deterministic=True)
            else:
                action, _ = self.policy.sample(state)

        action = action.cpu().numpy().flatten()

        # Ensure action is 2D
        if len(action) < 2:
            action = np.array([action[0] if len(action) > 0 else 0.0, 0.0])
        elif len(action) > 2:
            action = action[:2]

        action[0] = np.clip(action[0], 0.0, 15.0)
        action[1] = np.clip(action[1], -0.3, 0.3)

        return action

    def store_transition(self, obs, action, reward, next_obs, done, info):
        """
        Store transition in buffer with all data needed for risk target computation.
        """
        # Extract features
        state = self.extract_features(obs['images'], obs['ego_state'], obs.get('weather', 'clear'))
        next_state = self.extract_features(next_obs['images'], next_obs['ego_state'], next_obs.get('weather', 'clear'))

        # ========== FIX: Ensure state is 2D ==========
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(next_state.shape) == 1:
            next_state = next_state.reshape(1, -1)

        state_t = torch.FloatTensor(state).to(self.device)

        # ========== FIX: Ensure action is 2D ==========
        action = np.array(action).flatten()
        if len(action.shape) == 1:
            action = action.reshape(1, -1)
        action_t = torch.FloatTensor(action).to(self.device)

        with torch.no_grad():
            value = self.value(state_t).cpu().numpy()
            risk = self.risk_predictor(state_t, action_t).cpu().numpy()

        # ========== FIX: Store risk target computation data ==========
        collision = float(info.get('collision', False))
        speed = info.get('speed', 0.0)
        prev_action = self.buffer['actions'][-1] if len(self.buffer['actions']) > 0 else None
        weather_variance = info.get('weather_variance', 0.0)

        # Store (flattened)
        self.buffer['states'].append(state.flatten())
        self.buffer['actions'].append(action.flatten())
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state.flatten())
        self.buffer['dones'].append(float(done))
        self.buffer['values'].append(value.flatten()[0] if len(value.flatten()) > 0 else 0.0)
        self.buffer['risks'].append(risk.flatten()[0] if len(risk.flatten()) > 0 else 0.0)

        # Risk target data
        self.buffer['collisions'].append(collision)
        self.buffer['speeds'].append(speed)
        self.buffer['prev_actions'].append(prev_action)
        self.buffer['weather_variances'].append(weather_variance)

        self.total_steps += 1
    
    def compute_gae(self, rewards, values, dones, next_values):
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_risk_target(self, state, action, collisions, speeds, prev_actions, weather_variances):
        batch_size = state.shape[0]
        device = state.device
        
        collision_tensor = torch.FloatTensor(collisions).to(device)
        
        action_change_risk = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            if prev_actions[i] is not None:
                prev_action_t = torch.FloatTensor(prev_actions[i]).to(device)
                curr_action_t = action[i]
                diff = torch.norm(curr_action_t - prev_action_t)
                action_change_risk[i] = torch.clamp(diff / self.max_action_change, 0, 1)
        
        safety_risk = torch.max(collision_tensor, action_change_risk)
        
        speed_tensor = torch.FloatTensor(speeds).to(device)
        efficiency_risk = torch.clamp((self.ref_speed - speed_tensor) / self.ref_speed, 0, 1)
        
        weather_tensor = torch.FloatTensor(weather_variances).to(device)
        weather_risk = torch.clamp(weather_tensor / self.weather_variance_threshold, 0, 1)
        
        risk_target = (self.risk_safety_weight * safety_risk + 
                       self.risk_efficiency_weight * efficiency_risk + 
                       self.risk_weather_weight * weather_risk)
        
        return risk_target
    
    def update(self):
        if len(self.buffer['states']) < self.rollout_steps:
            return
        
        states = np.array(self.buffer['states'])
        actions = np.array(self.buffer['actions'])
        rewards = np.array(self.buffer['rewards'])
        next_states = np.array(self.buffer['next_states'])
        dones = np.array(self.buffer['dones'])
        old_values = np.array(self.buffer['values']).flatten()
        old_risks = np.array(self.buffer['risks']).flatten()
        
        collisions = self.buffer['collisions']
        speeds = self.buffer['speeds']
        prev_actions = self.buffer['prev_actions']
        weather_variances = self.buffer['weather_variances']
        
        next_values = []
        with torch.no_grad():
            for ns in next_states:
                ns_t = torch.FloatTensor(ns).to(self.device)
                nv = self.value(ns_t).cpu().numpy()
                next_values.append(nv)
        next_values = np.array(next_values).flatten()
        
        advantages, returns = self.compute_gae(rewards, old_values, dones, next_values)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        old_risks_t = torch.FloatTensor(old_risks).to(self.device)
        
        with torch.no_grad():
            old_dist = self.policy(states_t)
            old_log_probs = old_dist.log_prob(actions_t).sum(dim=-1)
        
        for _ in range(self.update_epochs):
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                idx = indices[start:end]
                
                batch_states = states_t[idx]
                batch_actions = actions_t[idx]
                batch_advantages = advantages_t[idx]
                batch_returns = returns_t[idx]
                batch_old_log_probs = old_log_probs[idx]
                
                dist = self.policy(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clip_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(ratio * batch_advantages, clip_ratio * batch_advantages).mean()
                
                entropy_loss = -self.entropy_coef * entropy
                
                current_risks = self.risk_predictor(batch_states, batch_actions)
                risk_loss = self.dual_lambda * torch.max(
                    current_risks.mean() - self.risk_threshold,
                    torch.tensor(0.0, device=self.device)
                )
                
                total_policy_loss = policy_loss + entropy_loss + risk_loss
                
                values = self.value(batch_states)
                value_loss = F.mse_loss(values, batch_returns)
                
                risk_pred = self.risk_predictor(batch_states, batch_actions)
                risk_target = self.compute_risk_target(
                    batch_states, 
                    batch_actions,
                    [collisions[i] for i in idx.tolist()],
                    [speeds[i] for i in idx.tolist()],
                    [prev_actions[i] for i in idx.tolist()],
                    [weather_variances[i] for i in idx.tolist()]
                )
                risk_loss_supervised = F.mse_loss(risk_pred, risk_target)
                
                self.policy_optimizer.zero_grad()
                total_policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                self.risk_optimizer.zero_grad()
                risk_loss_supervised.backward()
                torch.nn.utils.clip_grad_norm_(self.risk_predictor.parameters(), self.max_grad_norm)
                self.risk_optimizer.step()
        
        with torch.no_grad():
            avg_risk = old_risks_t.mean()
            self.dual_lambda = torch.clamp(
                self.dual_lambda + self.lr_dual * (avg_risk - self.risk_threshold),
                min=0.0
            )
        
        self.clear_buffer()
    
    def clear_buffer(self):
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'values': [],
            'risks': [],
            'collisions': [],
            'speeds': [],
            'prev_actions': [],
            'weather_variances': []
        }
    
    def get_checkpoint(self):
        return {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'risk_predictor_state_dict': self.risk_predictor.state_dict(),
            'perception_state_dict': self.perception.state_dict(),
            'dual_lambda': self.dual_lambda,
            'total_steps': self.total_steps
        }
    
    def load_checkpoint(self, checkpoint):
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.risk_predictor.load_state_dict(checkpoint['risk_predictor_state_dict'])
        self.perception.load_state_dict(checkpoint['perception_state_dict'])
        self.dual_lambda = checkpoint.get('dual_lambda', torch.tensor(0.0))
        self.total_steps = checkpoint.get('total_steps', 0)
        self.to_device()
    
    def set_weather(self, weather):
        self.perception.set_weather(weather)
    
    def get_perception_stats(self):
        return self.perception.get_weather_stats()
