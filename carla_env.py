"""
CARLA environment for unsignalized intersection autonomous driving.
"""

import os
import sys
import time
import random
import math
import numpy as np
from collections import deque

import carla
import cv2

try:
    import gym
    from gym import spaces
except ImportError:
    print('gym not installed, using custom environment wrapper')


class CarlaIntersectionEnv:
    """
    CARLA environment for unsignalized intersection driving tasks.
    Supports four tasks: straight, left turn, right turn, U-turn.
    """
    
    # Weather presets
    WEATHERS = {
        'clear': carla.WeatherParameters.ClearNoon,
        'rain': carla.WeatherParameters.HardRainNoon,
        'snow': carla.WeatherParameters.SnowNoon,
        'fog': carla.WeatherParameters.FoggyNoon,
        'glare': carla.WeatherParameters.SoftRainNoon,
        'heavy_rain': carla.WeatherParameters.HardRainNoon,
        'heavy_snow': carla.WeatherParameters.HardSnowNoon,
        'haze': carla.WeatherParameters.FoggyNoon
    }
    
    def __init__(self, config, host='localhost', port=6000):
        self.config = config
        self.host = host
        self.port = port
        
        # Camera settings
        self.img_width = config['camera']['img_width']
        self.img_height = config['camera']['img_height']
        self.fov = config['camera'].get('fov', 90)
        self.cam_x = config['camera'].get('cam_x', 2.5)
        self.cam_y = config['camera'].get('cam_y', 0.0)
        self.cam_z = config['camera'].get('cam_z', 3.2)
        self.cam_pitch = config['camera'].get('cam_pitch', -15)
        
        # Environment settings
        self.town = config['environment'].get('town', 'Town05')
        self.fps = config['environment'].get('fps', 10)
        self.max_episode_steps = config['environment'].get('max_episode_steps', 700)
        self.safe_distance = config['environment'].get('safe_distance', 5.0)
        self.lane_threshold = config['environment'].get('lane_threshold', 2.0)
        self.max_deviation_frames = config['environment'].get('max_deviation_frames', 20)
        
        # Reward weights
        self.safety_weight = config['reward'].get('safety_weight', 0.5)
        self.efficiency_weight = config['reward'].get('efficiency_weight', 0.3)
        self.weather_weight = config['reward'].get('weather_weight', 0.2)
        self.w_lane = config['reward'].get('w_lane', 0.1)
        self.w_risk_penalty = config['reward'].get('w_risk_penalty', 0.1)
        self.progress_scale = config['reward'].get('progress_scale', 1.0)
        self.ref_speed = config['reward'].get('ref_speed', 8.0)
        
        # Training weathers
        self.train_weathers = config['perception'].get('weathers', ['clear', 'rain', 'snow', 'fog'])
        
        # State
        self.stack_frames = config['perception'].get('stack_frames', 3)
        self.image_buffer = deque(maxlen=self.stack_frames)
        self.ego_state_buffer = deque(maxlen=self.stack_frames)
        
        # CARLA components
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.camera = None
        self.camera_rgb = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.traffic_manager = None
        
        # Spawn points
        self.ego_start_points = []
        self.ego_end_points = []
        self._setup_spawn_points()
        
        # Action space: [target_speed, steering_angle]
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.3]),
            high=np.array([15.0, 0.3]),
            dtype=np.float32
        )
        
        # State space (will be set later)
        self.state_dim = config['perception']['lstm_hidden_size']
        self.observation_space = spaces.Dict({
            'images': spaces.Box(
                low=0, high=255,
                shape=(self.stack_frames, 3, self.img_height, self.img_width),
                dtype=np.uint8
            ),
            'ego_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.stack_frames, 4),
                dtype=np.float32
            )
        })
        
        self.current_task = 'straight'
        self.current_weather = 'clear'
        self.episode_step = 0
        self.total_distance = 0.0
        self.prev_distance = 0.0
        self.collision_occurred = False
        self.lane_deviation_frames = 0
        
        # For feature variance tracking (weather robustness)
        self.feature_buffer = {}
    
    def _setup_spawn_points(self):
        """Setup spawn and end points for different tasks."""
        # Town05 intersection coordinates (approximate)
        # These should be adjusted based on actual map
        
        # Intersection center
        intersection_center = carla.Location(x=0, y=0, z=0.5)
        
        # Start points (one for each direction)
        start_points = [
            carla.Location(x=-20, y=0, z=0.5),    # South approach
            carla.Location(x=20, y=0, z=0.5),     # North approach
            carla.Location(x=0, y=-20, z=0.5),    # East approach
            carla.Location(x=0, y=20, z=0.5)      # West approach
        ]
        
        # End points (for different tasks)
        end_points = {
            'straight': [
                carla.Location(x=20, y=0, z=0.5),      # North exit
                carla.Location(x=-20, y=0, z=0.5),     # South exit
                carla.Location(x=0, y=20, z=0.5),      # East exit
                carla.Location(x=0, y=-20, z=0.5)      # West exit
            ],
            'left': [
                carla.Location(x=0, y=20, z=0.5),      # East exit
                carla.Location(x=0, y=-20, z=0.5),     # West exit
                carla.Location(x=-20, y=0, z=0.5),     # South exit
                carla.Location(x=20, y=0, z=0.5)       # North exit
            ],
            'right': [
                carla.Location(x=0, y=-20, z=0.5),     # West exit
                carla.Location(x=0, y=20, z=0.5),      # East exit
                carla.Location(x=20, y=0, z=0.5),      # North exit
                carla.Location(x=-20, y=0, z=0.5)      # South exit
            ],
            'uturn': [
                carla.Location(x=-20, y=0, z=0.5),     # South exit (turn around)
                carla.Location(x=20, y=0, z=0.5),      # North exit
                carla.Location(x=0, y=-20, z=0.5),     # West exit
                carla.Location(x=0, y=20, z=0.5)       # East exit
            ]
        }
        
        self.ego_start_points = start_points
        self.ego_end_points = end_points
    
    def setup(self):
        """Initialize CARLA environment."""
        # Connect to CARLA server
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.town)
        
        # Setup traffic manager
        self.traffic_manager = TrafficManager(self.client, self.world)
        
        # Spawn ego vehicle
        self._spawn_ego_vehicle()
        
        # Setup sensors
        self._setup_camera()
        self._setup_collision_sensor()
        self._setup_lane_sensor()
        
        # Set spectator
        self._set_spectator()
        
        # Spawn traffic vehicles
        self.traffic_manager.spawn_traffic()
        
        print('Environment setup completed')
    
    def _spawn_ego_vehicle(self):
        """Spawn ego vehicle."""
        blueprint_library = self.world.get_blueprint_library()
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Random spawn point
        spawn_index = random.randint(0, len(self.ego_start_points) - 1)
        spawn_point = self.ego_start_points[spawn_index]
        
        # Add rotation
        rotation = carla.Rotation(yaw=spawn_index * 90)
        spawn_point = carla.Transform(spawn_point, rotation)
        
        self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
        self.ego_start_index = spawn_index
    
    def _setup_camera(self):
        """Setup front-facing RGB camera."""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.img_width))
        camera_bp.set_attribute('image_size_y', str(self.img_height))
        camera_bp.set_attribute('fov', str(self.fov))
        
        camera_transform = carla.Transform(
            carla.Location(x=self.cam_x, y=self.cam_y, z=self.cam_z),
            carla.Rotation(pitch=self.cam_pitch)
        )
        
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.camera.listen(lambda image: self._process_image(image))
        
        # Initialize image buffer
        self.image_buffer.append(np.zeros((3, self.img_height, self.img_width), dtype=np.uint8))
    
    def _process_image(self, image):
        """Process camera image."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.img_height, self.img_width, 4))
        rgb = array[:, :, :3]
        rgb = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
        
        self.image_buffer.append(rgb)
    
    def _setup_collision_sensor(self):
        """Setup collision sensor."""
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_bp.set_attribute('only_actors', 'vehicle.*')
        
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))
    
    def _on_collision(self, event):
        """Handle collision event."""
        self.collision_occurred = True
    
    def _setup_lane_sensor(self):
        """Setup lane invasion sensor."""
        blueprint_library = self.world.get_blueprint_library()
        lane_bp = blueprint_library.find('sensor.other.lane_invasion')
        
        self.lane_sensor = self.world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.lane_sensor.listen(lambda event: self._on_lane_invasion(event))
    
    def _on_lane_invasion(self, event):
        """Handle lane invasion event."""
        self.lane_deviation_frames += 1
    
    def _set_spectator(self):
        """Set spectator camera to follow ego vehicle."""
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            carla.Location(x=transform.location.x - 10, 
                          y=transform.location.y, 
                          z=transform.location.z + 10),
            carla.Rotation(pitch=-30, yaw=transform.rotation.yaw)
        ))
    
    def set_weather(self, weather):
        """Set weather condition."""
        if weather in self.WEATHERS:
            self.current_weather = weather
            self.world.set_weather(self.WEATHERS[weather])
    
    def reset(self):
        """Reset environment for new episode."""
        # Clear buffers
        self.image_buffer.clear()
        self.ego_state_buffer.clear()
        self.collision_occurred = False
        self.lane_deviation_frames = 0
        self.episode_step = 0
        self.total_distance = 0.0
        self.prev_distance = 0.0
        
        # Reset ego vehicle
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
        
        # Spawn ego vehicle
        self._spawn_ego_vehicle()
        
        # Reset sensors
        if self.camera is not None:
            self.camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_sensor is not None:
            self.lane_sensor.destroy()
        
        # Setup sensors
        self._setup_camera()
        self._setup_collision_sensor()
        self._setup_lane_sensor()
        
        # Set weather (randomly select from training weathers)
        weather = random.choice(self.train_weathers)
        self.set_weather(weather)
        
        # Set task (randomly select)
        self.current_task = random.choice(['straight', 'left', 'right', 'uturn'])
        
        # Reset traffic
        self.traffic_manager.reset()
        
        # Wait for initialization
        time.sleep(0.5)
        
        # Get initial observation
        obs = self._get_observation()
        self.prev_distance = self._get_distance_to_goal()
        
        return obs
    
    def _get_observation(self):
        """Get current observation."""
        # Get current images
        images = np.array(list(self.image_buffer))
        if len(images) < self.stack_frames:
            # Pad with zeros if buffer not full
            pad = self.stack_frames - len(images)
            zeros = np.zeros((pad, 3, self.img_height, self.img_width), dtype=np.uint8)
            images = np.concatenate([zeros, images], axis=0)
        
        # Get ego state
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        ego_state = np.array([
            speed,
            transform.location.x,
            transform.location.y,
            np.deg2rad(transform.rotation.yaw)
        ], dtype=np.float32)
        
        self.ego_state_buffer.append(ego_state)
        ego_states = np.array(list(self.ego_state_buffer))
        if len(ego_states) < self.stack_frames:
            pad = self.stack_frames - len(ego_states)
            zeros = np.zeros((pad, 4), dtype=np.float32)
            ego_states = np.concatenate([zeros, ego_states], axis=0)
        
        return {
            'images': images,
            'ego_state': ego_states,
            'weather': self.current_weather
        }
    
    def _get_distance_to_goal(self):
        """Get distance to goal."""
        transform = self.ego_vehicle.get_transform()
        end_points = self.ego_end_points.get(self.current_task, [])
        if end_points and len(end_points) > self.ego_start_index:
            goal = end_points[self.ego_start_index]
            dx = transform.location.x - goal.x
            dy = transform.location.y - goal.y
            return np.sqrt(dx**2 + dy**2)
        return 100.0
    
    def _is_at_goal(self):
        """Check if ego vehicle reached goal."""
        distance = self._get_distance_to_goal()
        return distance < 3.0
    
    def _is_off_road(self):
        """Check if ego vehicle is off-road."""
        # Simplified: check if vehicle is far from lane center
        # In practice, use waypoint API
        return self.lane_deviation_frames > self.max_deviation_frames
    
    def _compute_reward(self, action, info):
        """Compute reward based on safety, efficiency, and weather robustness."""
        # Get current state
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Current distance to goal
        current_distance = self._get_distance_to_goal()
        
        # Safety reward
        # Distance to nearest obstacle
        min_distance = self._get_min_distance_to_obstacle()
        safety_reward = np.exp(-self.safe_distance / (min_distance + 1e-6))
        safety_reward = self.safety_weight * safety_reward
        
        # Collision penalty
        if self.collision_occurred:
            safety_reward -= 10.0
        
        # Efficiency reward
        # Progress
        progress = (self.prev_distance - current_distance) / 10.0
        efficiency_reward = self.efficiency_weight * max(progress, -1.0)
        
        # Speed reward
        speed_reward = speed / self.ref_speed
        efficiency_reward += self.efficiency_weight * 0.3 * speed_reward
        
        # Smoothness reward (penalize abrupt actions)
        # (simplified, in practice need to store previous action)
        smoothness_reward = 0.0
        
        # Weather robustness reward
        # Feature variance across weathers
        weather_reward = self.weather_weight * self._compute_weather_robustness_reward()
        
        # Lane keeping penalty
        lane_penalty = -self.w_lane * (self.lane_deviation_frames / self.max_episode_steps)
        
        # Risk penalty
        risk_penalty = -self.w_risk_penalty * info.get('risk', 0.0)
        
        # Total reward
        reward = (safety_reward + efficiency_reward + weather_reward + 
                  smoothness_reward + lane_penalty + risk_penalty)
        
        # Goal completion bonus
        if self._is_at_goal():
            reward += 20.0
        
        self.prev_distance = current_distance
        
        return reward
    
    def _get_min_distance_to_obstacle(self):
        """Get minimum distance to nearest obstacle."""
        min_dist = float('inf')
        transform = self.ego_vehicle.get_transform()
        
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.id == self.ego_vehicle.id:
                continue
            
            other_transform = vehicle.get_transform()
            dx = transform.location.x - other_transform.location.x
            dy = transform.location.y - other_transform.location.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
        
        return min_dist if min_dist != float('inf') else 50.0
    
    def _compute_weather_robustness_reward(self):
        """Compute weather robustness reward based on feature variance."""
        # In practice, this should use actual feature variance
        # For now, return a constant
        return 0.1
    
    def _apply_control(self, action):
        """Apply control action to ego vehicle."""
        target_speed = float(action[0])
        steering = float(action[1])
        
        # Clamp values
        target_speed = np.clip(target_speed, 0.0, 15.0)
        steering = np.clip(steering, -0.3, 0.3)
        
        # Convert to CARLA control
        control = carla.VehicleControl()
        
        # Simple PID for speed control (simplified)
        velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        speed_error = target_speed - current_speed
        control.throttle = np.clip(0.2 * speed_error + 0.3, 0.0, 1.0)
        control.brake = np.clip(-0.2 * speed_error, 0.0, 1.0) if speed_error < 0 else 0.0
        control.steer = steering
        
        self.ego_vehicle.apply_control(control)
    
    def step(self, action):
        """Execute one step in the environment."""
        self.episode_step += 1
        
        # Apply action
        self._apply_control(action)
        
        # Tick world
        self.world.tick()
        
        # Update spectator
        self._set_spectator()
        
        # Get observation
        obs = self._get_observation()
        
        # Check termination conditions
        done = False
        info = {}
        
        # Collision
        if self.collision_occurred:
            done = True
            info['collision'] = True
            info['fail_type'] = 1
        
        # Timeout
        if self.episode_step >= self.max_episode_steps:
            done = True
            info['timeout'] = True
            info['fail_type'] = 2
        
        # Goal reached
        if self._is_at_goal():
            done = True
            info['success'] = True
            info['fail_type'] = 3
        
        # Off-road
        if self._is_off_road():
            done = True
            info['off_road'] = True
            info['fail_type'] = 4
        
        # Compute reward
        reward = self._compute_reward(action, info)
        
        # Gather info
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        current_distance = self._get_distance_to_goal()
        
        info.update({
            'speed': speed,
            'distance_to_goal': current_distance,
            'steps': self.episode_step,
            'weather': self.current_weather,
            'task': self.current_task
        })
        
        return obs, reward, done, info
    
    def close(self):
        """Close environment."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        # Destroy ego vehicle
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
        
        # Destroy sensors
        for sensor in [self.camera, self.collision_sensor, self.lane_sensor]:
            if sensor is not None:
                sensor.destroy()
        
        # Cleanup traffic
        if self.traffic_manager is not None:
            self.traffic_manager.cleanup()
        
        print('Environment cleanup completed')
