"""
CARLA environment for unsignalized intersection autonomous driving.
Supports four separate tasks: straight, left turn, right turn, U-turn.
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

# Import TrafficManager from separate file
from traffic_manager import TrafficManager


class CarlaIntersectionEnv:
    """
    CARLA environment for unsignalized intersection driving tasks.
    Supports four independent tasks: straight, left turn, right turn, U-turn.
    Each task has its own fixed start and end positions.
    """
    
    def __init__(self, config, host='localhost', port=6000, task='straight'):
        """
        Initialize environment.
        
        Args:
            config: Configuration dictionary
            host: CARLA server host
            port: CARLA server port
            task: One of 'straight', 'left', 'right', 'uturn'
        """
        self.config = config
        self.host = host
        self.port = port
        self.task = task  # Current task
        
        # Validate task
        if task not in ['straight', 'left', 'right', 'uturn']:
            raise ValueError(f"Task '{task}' not found. Available: straight, left, right, uturn")
        
        # ========== Task Configurations ==========
        self.TASK_CONFIGS = {
            'straight': {
                'start': carla.Transform(
                    carla.Location(x=28.0, y=73.0, z=0.5),
                    carla.Rotation(yaw=90.0)
                ),
                'end': carla.Location(x=28.0, y=105.0, z=0.5),
                'description': 'Go straight through intersection'
            },
            'left': {
                'start': carla.Transform(
                    carla.Location(x=28.0, y=73.0, z=0.5),
                    carla.Rotation(yaw=90.0)
                ),
                'end': carla.Location(x=45.0, y=94.0, z=0.5),
                'description': 'Turn left at intersection'
            },
            'right': {
                'start': carla.Transform(
                    carla.Location(x=28.0, y=73.0, z=0.5),
                    carla.Rotation(yaw=90.0)
                ),
                'end': carla.Location(x=14.0, y=88.0, z=0.5),
                'description': 'Turn right at intersection'
            },
            'uturn': {
                'start': carla.Transform(
                    carla.Location(x=35.0, y=73.0, z=0.5),
                    carla.Rotation(yaw=270.0)
                ),
                'end': carla.Location(x=35.0, y=73.0, z=0.5),
                'description': 'Make a U-turn at intersection'
            }
        }
        
        # Get task configuration
        self.task_config = self.TASK_CONFIGS[task]
        self.start_transform = self.task_config['start']
        self.end_location = self.task_config['end']
        
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
        self.train_weathers = config['perception'].get('weathers', ['clear', 'light_rain', 'light_snow', 'light_fog'])
        
        # State
        self.stack_frames = config['perception'].get('stack_frames', 3)
        self.image_buffer = deque(maxlen=self.stack_frames)
        self.ego_state_buffer = deque(maxlen=self.stack_frames)
        
        # CARLA components
        self.client = None
        self.world = None
        self.map = None
        self.ego_vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None
        
        # Traffic Manager (from separate file)
        self.traffic_manager = None
        
        # Action space: [target_speed, steering_angle]
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.3], dtype=np.float32),
            high=np.array([15.0, 0.3], dtype=np.float32),
            dtype=np.float32
        )
        
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
        
        self.current_task = task
        self.current_weather = 'clear'
        self.episode_step = 0
        self.total_distance = 0.0
        self.prev_distance = 0.0
        self.collision_occurred = False
        self.lane_deviation_frames = 0
        
        self.feature_buffer = {}
        self.weather_history = []
        self.current_image = None
        self.image_received = False
        
        # Fixed spectator position (俯视路口)
        self.spectator_location = carla.Location(x=25.0, y=90.0, z=50.0)
        self.spectator_rotation = carla.Rotation(pitch=-90.0, yaw=90.0, roll=0.0)
        
        print(f'Environment initialized for task: {task}')
        print(f'  Start: ({self.start_transform.location.x}, {self.start_transform.location.y})')
        print(f'  End: ({self.end_location.x}, {self.end_location.y})')
    
    # ============================================================
    # WEATHER PARAMETERS (Based on your paper's Table)
    # ============================================================
    def _create_weather_parameters(self, weather_name):
        """
        Create CARLA weather parameters based on the paper's Table.
        """
        if weather_name == 'clear':
            return carla.WeatherParameters(
                cloudiness=0.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=10.0,
                sun_azimuth_angle=90.0,
                sun_altitude_angle=45.0,
                fog_density=0.0,
                fog_distance=0.0,
                wetness=0.0,
                fog_falloff=0.0
            )
        elif weather_name == 'light_rain':
            return carla.WeatherParameters(
                cloudiness=60.0,
                precipitation=20.0,
                precipitation_deposits=20.0,
                wind_intensity=30.0,
                sun_azimuth_angle=90.0,
                sun_altitude_angle=45.0,
                fog_density=0.0,
                fog_distance=0.0,
                wetness=60.0,
                fog_falloff=0.0
            )
        elif weather_name == 'light_snow':
            return carla.WeatherParameters(
                cloudiness=80.0,
                precipitation=10.0,
                precipitation_deposits=50.0,
                wind_intensity=20.0,
                sun_azimuth_angle=90.0,
                sun_altitude_angle=30.0,
                fog_density=0.0,
                fog_distance=0.0,
                wetness=60.0,
                fog_falloff=0.0
            )
        elif weather_name == 'light_fog':
            return carla.WeatherParameters(
                cloudiness=20.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=10.0,
                sun_azimuth_angle=90.0,
                sun_altitude_angle=45.0,
                fog_density=40.0,
                fog_distance=100.0,
                wetness=0.0,
                fog_falloff=0.3
            )
        elif weather_name == 'glare':
            return carla.WeatherParameters(
                cloudiness=0.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=5.0,
                sun_azimuth_angle=90.0,
                sun_altitude_angle=10.0,  # Low angle for glare
                fog_density=0.0,
                fog_distance=0.0,
                wetness=0.0,
                fog_falloff=0.0
            )
        elif weather_name == 'heavy_rain':
            return carla.WeatherParameters(
                cloudiness=100.0,
                precipitation=100.0,
                precipitation_deposits=100.0,
                wind_intensity=60.0,
                sun_azimuth_angle=90.0,
                sun_altitude_angle=20.0,
                fog_density=0.0,
                fog_distance=0.0,
                wetness=100.0,
                fog_falloff=0.0
            )
        elif weather_name == 'heavy_snow':
            return carla.WeatherParameters(
                cloudiness=100.0,
                precipitation=50.0,
                precipitation_deposits=100.0,
                wind_intensity=40.0,
                sun_azimuth_angle=90.0,
                sun_altitude_angle=15.0,
                fog_density=10.0,
                fog_distance=80.0,
                wetness=100.0,
                fog_falloff=0.2
            )
        elif weather_name == 'haze':
            return carla.WeatherParameters(
                cloudiness=80.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=15.0,
                sun_azimuth_angle=90.0,
                sun_altitude_angle=30.0,
                fog_density=60.0,
                fog_distance=60.0,
                wetness=80.0,
                fog_falloff=0.4
            )
        else:
            print(f'Weather {weather_name} not recognized, using clear')
            return carla.WeatherParameters.ClearNoon
    
    def set_weather(self, weather_name):
        """Set weather condition using the paper's parameters."""
        self.current_weather = weather_name
        weather_params = self._create_weather_parameters(weather_name)
        if weather_params is not None:
            self.world.set_weather(weather_params)
            self.world.tick()
    
    def get_weather_description(self, weather_name):
        """Get weather description for logging."""
        weather_descriptions = {
            'clear': 'Cloud:0%, Precip:0%, Fog:0%, Wetness:0%, Sun Alt:45°',
            'light_rain': 'Cloud:60%, Precip:20%, Fog:0%, Wetness:60%, Sun Alt:45°',
            'light_snow': 'Cloud:80%, Precip:10%, Fog:0%, Wetness:60%, Sun Alt:30°',
            'light_fog': 'Cloud:20%, Precip:0%, Fog:40%, Wetness:0%, Sun Alt:45°',
            'glare': 'Cloud:0%, Precip:0%, Fog:0%, Wetness:0%, Sun Alt:10°',
            'heavy_rain': 'Cloud:100%, Precip:100%, Fog:0%, Wetness:100%, Sun Alt:20°',
            'heavy_snow': 'Cloud:100%, Precip:50%, Fog:10%, Wetness:100%, Sun Alt:15°',
            'haze': 'Cloud:80%, Precip:0%, Fog:60%, Wetness:80%, Sun Alt:30°'
        }
        return weather_descriptions.get(weather_name, 'Unknown weather')
    
    # ============================================================
    # ENVIRONMENT SETUP
    # ============================================================
    def setup(self):
        """Initialize CARLA environment."""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.load_world(self.town)
            self.map = self.world.get_map()
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.fps
            self.world.apply_settings(settings)
            
            # Initialize Traffic Manager (from separate file)
            self.traffic_manager = TrafficManager(self.client, self.world)
            
            # Spawn ego vehicle at task-specific start position
            self._spawn_ego_vehicle()
            
            # Setup sensors
            self._setup_camera()
            self._setup_collision_sensor()
            self._setup_lane_sensor()
            
            # Set fixed spectator view (俯视路口)
            self._set_fixed_spectator()
            
            # Spawn traffic vehicles (uses your spawn_vehicle method)
            self.traffic_manager.spawn_traffic(use_fixed_positions=True)
            
            # Initialize buffers
            self._initialize_buffers()
            
            print(f'Environment setup completed for task: {self.task}')
            print(f'  Weather: {self.current_weather}')
            print(f'  Start: ({self.start_transform.location.x}, {self.start_transform.location.y})')
            print(f'  End: ({self.end_location.x}, {self.end_location.y})')
            
        except Exception as e:
            print(f'Error during setup: {e}')
            self.cleanup()
            raise
    
    def _initialize_buffers(self):
        """Initialize image and state buffers."""
        for _ in range(self.stack_frames):
            self.image_buffer.append(np.zeros((3, self.img_height, self.img_width), dtype=np.uint8))
            self.ego_state_buffer.append(np.zeros(4, dtype=np.float32))

    def _spawn_ego_vehicle(self):
        """Spawn ego vehicle at task-specific start position with retry logic."""
        blueprint_library = self.world.get_blueprint_library()
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'ego')

        start_transform = self.start_transform

        # Try to spawn with position offsets
        for attempt in range(10):
            offset_x = (attempt % 3) * 0.2
            offset_y = (attempt // 3) * 0.2
            adjusted_loc = carla.Location(
                x=start_transform.location.x + offset_x,
                y=start_transform.location.y + offset_y,
                z=start_transform.location.z
            )
            adjusted_transform = carla.Transform(adjusted_loc, start_transform.rotation)

            self.ego_vehicle = self.world.try_spawn_actor(ego_bp, adjusted_transform)
            if self.ego_vehicle is not None:
                print(f'Ego vehicle spawned at ({adjusted_loc.x:.1f}, {adjusted_loc.y:.1f})')
                return

            self.world.tick()

        # Final attempt at original position
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, start_transform)
        if self.ego_vehicle is None:
            raise RuntimeError(f'Failed to spawn ego vehicle for task {self.task}')

        print(f'Ego vehicle spawned at ({start_transform.location.x}, {start_transform.location.y})')



    def _set_fixed_spectator(self):
        """Set fixed spectator view for training visualization."""
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(
            self.spectator_location,
            self.spectator_rotation
        ))
    
    # ============================================================
    # SENSOR SETUP
    # ============================================================
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
        
        self.image_received = False
        start_time = time.time()
        while not self.image_received and time.time() - start_time < 5.0:
            self.world.tick()
    
    def _process_image(self, image):
        """Process camera image."""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((self.img_height, self.img_width, 4))
            rgb = array[:, :, :3]
            rgb = np.transpose(rgb, (2, 0, 1))
            self.current_image = rgb
            self.image_received = True
        except Exception as e:
            print(f'Error processing image: {e}')

    def _setup_collision_sensor(self):
        """Setup collision sensor."""
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        # 移除 only_actors 属性设置（CARLA 0.9.10 不支持此属性）
        # collision_bp.set_attribute('only_actors', 'vehicle.*')

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
    
    # ============================================================
    # OBSERVATION AND STATE
    # ============================================================
    def _get_current_image(self):
        """Get current image from buffer."""
        if self.current_image is not None and self.image_received:
            return self.current_image
        return np.zeros((3, self.img_height, self.img_width), dtype=np.uint8)
    
    def _get_ego_state(self):
        """Get current ego vehicle state."""
        if self.ego_vehicle is None:
            return np.zeros(4, dtype=np.float32)
        
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        yaw = math.radians(transform.rotation.yaw)
        
        return np.array([speed, transform.location.x, transform.location.y, yaw], dtype=np.float32)
    
    def _get_observation(self):
        """Get current observation."""
        image = self._get_current_image()
        self.image_buffer.append(image)
        
        ego_state = self._get_ego_state()
        self.ego_state_buffer.append(ego_state)
        
        images = np.array(list(self.image_buffer))
        if len(images) < self.stack_frames:
            pad = self.stack_frames - len(images)
            zeros = np.zeros((pad, 3, self.img_height, self.img_width), dtype=np.uint8)
            images = np.concatenate([zeros, images], axis=0)
        
        ego_states = np.array(list(self.ego_state_buffer))
        if len(ego_states) < self.stack_frames:
            pad = self.stack_frames - len(ego_states)
            zeros = np.zeros((pad, 4), dtype=np.float32)
            ego_states = np.concatenate([zeros, ego_states], axis=0)
        
        return {
            'images': images,
            'ego_state': ego_states,
            'weather': self.current_weather,
            'task': self.current_task
        }
    
    def _get_distance_to_goal(self):
        """Get distance to goal (Euclidean distance to end location)."""
        if self.ego_vehicle is None:
            return 100.0
        
        transform = self.ego_vehicle.get_transform()
        dx = transform.location.x - self.end_location.x
        dy = transform.location.y - self.end_location.y
        return np.sqrt(dx**2 + dy**2)
    
    def _is_at_goal(self):
        """Check if ego vehicle reached goal."""
        return self._get_distance_to_goal() < 3.0
    
    def _is_off_road(self):
        """Check if ego vehicle is off-road."""
        return self.lane_deviation_frames > self.max_deviation_frames
    
    def _get_min_distance_to_obstacle(self):
        """Get minimum distance to nearest obstacle."""
        if self.ego_vehicle is None:
            return 50.0
        
        min_dist = 50.0
        transform = self.ego_vehicle.get_transform()
        
        all_vehicles = self.world.get_actors().filter('vehicle.*')
        for vehicle in all_vehicles:
            if vehicle.id == self.ego_vehicle.id:
                continue
            
            other_transform = vehicle.get_transform()
            dx = transform.location.x - other_transform.location.x
            dy = transform.location.y - other_transform.location.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def _compute_weather_robustness_reward(self):
        """Compute weather robustness reward."""
        if len(self.weather_history) < 2:
            return 0.0
        
        states = []
        for w in self.weather_history[-4:]:
            states.append(self._get_ego_state())
        
        if len(states) >= 2:
            states = np.array(states)
            variance = np.var(states, axis=0).mean()
            return max(0.0, 1.0 - variance / 10.0)
        
        return 0.0
    
    # ============================================================
    # REWARD AND CONTROL
    # ============================================================
    def _compute_reward(self, action, info):
        """Compute reward based on safety, efficiency, and weather robustness."""
        if self.ego_vehicle is None:
            return 0.0
        
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        current_distance = self._get_distance_to_goal()
        
        # Safety Reward
        min_distance = self._get_min_distance_to_obstacle()
        safety_reward = np.exp(-self.safe_distance / (min_distance + 1e-6))
        
        if self.collision_occurred:
            safety_reward = -10.0
        else:
            safety_reward = self.safety_weight * safety_reward
        
        # Efficiency Reward
        progress = (self.prev_distance - current_distance) / 10.0
        efficiency_reward = self.efficiency_weight * max(progress, -1.0)
        speed_reward = min(speed / self.ref_speed, 1.0)
        efficiency_reward += self.efficiency_weight * 0.3 * speed_reward
        
        # Weather Robustness Reward
        weather_reward = self.weather_weight * self._compute_weather_robustness_reward()
        
        # Penalties
        lane_penalty = -self.w_lane * (self.lane_deviation_frames / self.max_episode_steps)
        risk = info.get('risk', 0.0)
        risk_penalty = -self.w_risk_penalty * risk
        
        # Total Reward
        reward = safety_reward + efficiency_reward + weather_reward + lane_penalty + risk_penalty
        
        if self._is_at_goal():
            reward += 20.0
        
        self.prev_distance = current_distance
        return reward
    
    def _apply_control(self, action):
        """Apply control action to ego vehicle."""
        if self.ego_vehicle is None:
            return
        
        target_speed = np.clip(float(action[0]), 0.0, 15.0)
        steering = np.clip(float(action[1]), -0.3, 0.3)
        
        velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        speed_error = target_speed - current_speed
        throttle = np.clip(0.5 * speed_error + 0.2, 0.0, 1.0)
        brake = np.clip(-0.5 * speed_error, 0.0, 1.0) if speed_error < 0 else 0.0
        
        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.steer = steering
        
        self.ego_vehicle.apply_control(control)
    
    # ============================================================
    # RESET AND STEP
    # ============================================================
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
        self.image_received = False

        # Destroy existing ego vehicle and sensors
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

        for sensor in [self.camera, self.collision_sensor, self.lane_sensor]:
            if sensor is not None:
                sensor.destroy()
                sensor = None

        # Reset traffic manager
        if self.traffic_manager is not None:
            self.traffic_manager.reset()

        # Spawn new ego vehicle at task-specific start
        self._spawn_ego_vehicle()

        # Setup sensors
        self._setup_camera()
        self._setup_collision_sensor()
        self._setup_lane_sensor()

        # Set fixed spectator view
        self._set_fixed_spectator()

        # Set weather (randomly select from training weathers)
        weather = random.choice(self.train_weathers)
        self.set_weather(weather)

        # Get initial observation
        self.world.tick()
        self._initialize_buffers()

        obs = self._get_observation()
        self.prev_distance = self._get_distance_to_goal()

        self.weather_history.append(weather)
        if len(self.weather_history) > 10:
            self.weather_history.pop(0)

        return obs
    
    def step(self, action):
        """Execute one step in the environment."""
        self.episode_step += 1
        
        self._apply_control(action)
        self.world.tick()
        
        obs = self._get_observation()
        
        done = False
        info = {
            'weather': self.current_weather,
            'task': self.current_task,
            'steps': self.episode_step
        }
        
        # Collision
        if self.collision_occurred:
            done = True
            info['collision'] = True
            info['fail_type'] = 1
            info['success'] = False
        
        # Timeout
        if self.episode_step >= self.max_episode_steps:
            done = True
            info['timeout'] = True
            info['fail_type'] = 2
            info['success'] = False
        
        # Goal reached
        if self._is_at_goal():
            done = True
            info['success'] = True
            info['fail_type'] = 3
            info['completion_time'] = self.episode_step / self.fps
        
        # Off-road
        if self._is_off_road():
            done = True
            info['off_road'] = True
            info['fail_type'] = 1
            info['success'] = False
        
        reward = self._compute_reward(action, info)
        
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        info.update({
            'speed': speed,
            'distance_to_goal': self._get_distance_to_goal(),
            'avg_speed': speed if done else 0.0,
            'drivew_score': self._compute_weather_robustness_reward()
        })
        
        return obs, reward, done, info
    
    # ============================================================
    # TASK MANAGEMENT
    # ============================================================
    def set_task(self, task):
        """Switch to a different task."""
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Task '{task}' not found. Available: {list(self.TASK_CONFIGS.keys())}")
        
        self.task = task
        self.task_config = self.TASK_CONFIGS[task]
        self.start_transform = self.task_config['start']
        self.end_location = self.task_config['end']
        self.current_task = task
        
        print(f'Switched to task: {task}')
        print(f'  Start: ({self.start_transform.location.x}, {self.start_transform.location.y})')
        print(f'  End: ({self.end_location.x}, {self.end_location.y})')
    
    def get_task_info(self):
        """Get information about current task."""
        return {
            'task': self.current_task,
            'start': {
                'x': self.start_transform.location.x,
                'y': self.start_transform.location.y,
                'yaw': self.start_transform.rotation.yaw
            },
            'end': {
                'x': self.end_location.x,
                'y': self.end_location.y
            },
            'description': self.task_config['description']
        }
    
    # ============================================================
    # CLEANUP
    # ============================================================
    def close(self):
        """Close environment."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up all resources."""
        print(f'Cleaning up environment for task: {self.task}')
        
        for sensor in [self.camera, self.collision_sensor, self.lane_sensor]:
            if sensor is not None:
                try:
                    sensor.destroy()
                except Exception as e:
                    print(f'Error destroying sensor: {e}')
        
        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.destroy()
            except Exception as e:
                print(f'Error destroying ego vehicle: {e}')
        
        if self.traffic_manager is not None:
            try:
                self.traffic_manager.cleanup()
            except Exception as e:
                print(f'Error cleaning traffic manager: {e}')
        
        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except Exception as e:
                print(f'Error resetting world settings: {e}')
        
        self.ego_vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.traffic_manager = None
        
        print('Environment cleanup completed')
