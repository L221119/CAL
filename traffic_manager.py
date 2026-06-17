"""
Traffic Manager for spawning and controlling background vehicles.
"""

import random
import carla
import time


class TrafficManager:
    """
    Manages background traffic vehicles for intersection scenarios.
    """

    def __init__(self, client, world, num_vehicles=8):
        self.client = client
        self.world = world
        self.num_vehicles = num_vehicles
        self.vehicles = []
        self.tm = None
        self.map = world.get_map()

        # Vehicle blueprint
        self.vehicle_bps = world.get_blueprint_library()
        self.auto_vehicle_list = []
        self.actor_list = []

    def _setup_spawn_points(self):
        """Setup spawn points for traffic vehicles."""
        all_spawn_points = self.world.get_map().get_spawn_points()

        intersection_center = carla.Location(x=0, y=0, z=0)
        near_intersection = []

        for sp in all_spawn_points:
            dx = sp.location.x - intersection_center.x
            dy = sp.location.y - intersection_center.y
            distance = (dx**2 + dy**2)**0.5

            if 30 < distance < 100:
                near_intersection.append(sp)

        if len(near_intersection) < self.num_vehicles:
            near_intersection = all_spawn_points

        self.spawn_points = near_intersection

    def spawn_vehicle(self):
        """
        Spawn background traffic vehicles at fixed positions.
        """
        # Clear existing vehicles
        self.cleanup()

        # Wait a moment for cleanup to complete
        self.world.tick()
        time.sleep(0.3)

        # Get vehicle blueprint
        auto_vehicle_bp = self.vehicle_bps.find('vehicle.audi.a2')

        # Spawn vehicles at specific locations
        location = carla.Location()
        location.z = 0.5

        spawned_count = 0

        # 辅助函数：尝试生成车辆，失败不报错
        def try_spawn(loc):
            nonlocal spawned_count
            auto_waypoint = self.map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if auto_waypoint is not None:
                # 使用 try_spawn_actor，不会因碰撞报错
                auto_vehicle = self.world.try_spawn_actor(auto_vehicle_bp, auto_waypoint.transform)
                if auto_vehicle is not None:
                    auto_vehicle.set_autopilot(True, 8000)
                    self.auto_vehicle_list.append(auto_vehicle)
                    self.actor_list.append(auto_vehicle)
                    self.vehicles.append(auto_vehicle)
                    spawned_count += 1
                    return True
            return False

        # Group 1: x=14, y=92, 2 vehicles
        location.x = 14
        location.y = 92
        for i in range(2):
            try_spawn(location)
            location.x = location.x - 8

        # Group 2: x=14, y=95, 1 vehicle
        location.x = 14
        location.y = 95
        for i in range(1):
            try_spawn(location)
            location.x = location.x - 8

        # Group 3: x=45, y=88, 1 vehicle
        location.x = 45
        location.y = 88
        for i in range(1):
            try_spawn(location)
            location.x = location.x + 8

        # Group 4: x=45, y=85, 2 vehicles
        location.x = 45
        location.y = 85
        for i in range(2):
            try_spawn(location)
            location.x = location.x + 8

        # Group 5: x=35, y=105, 1 vehicle
        location.x = 35
        location.y = 105
        for i in range(1):
            try_spawn(location)
            location.y = location.y + 5

        print(f'Spawned {spawned_count} traffic vehicles at fixed positions')
        return spawned_count

    def spawn_traffic(self, use_fixed_positions=True):
        """
        Spawn background traffic vehicles.

        Args:
            use_fixed_positions: If True, use fixed positions from spawn_vehicle().
                                If False, use random spawn points.
        """
        if use_fixed_positions:
            return self.spawn_vehicle()
        else:
            return self._spawn_random_traffic()

    def _spawn_random_traffic(self):
        """Spawn traffic at random spawn points."""
        if self.tm is None:
            self.tm = self.client.get_trafficmanager(8000)
            self.tm.set_global_distance_to_leading_vehicle(3.0)
            self.tm.set_synchronous_mode(True)

        selected_points = random.sample(
            self.spawn_points,
            min(self.num_vehicles, len(self.spawn_points))
        )

        blueprint_library = self.world.get_blueprint_library()
        spawned_count = 0

        for spawn_point in selected_points:
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if vehicle is not None:
                self.vehicles.append(vehicle)
                destination = random.choice(self.spawn_points)
                self.tm.set_destination(vehicle, destination.location)
                speed = random.uniform(5.0, 12.0)
                self.tm.set_path(vehicle, [destination])
                self.tm.set_vehicle_speed(vehicle, speed)
                spawned_count += 1

        print(f'Spawned {spawned_count} random traffic vehicles')
        return spawned_count

    def reset(self):
        """Reset traffic manager."""
        self.cleanup()
        # 等待车辆完全销毁
        self.world.tick()
        time.sleep(0.3)
        self.spawn_traffic(use_fixed_positions=True)

    def cleanup(self):
        """Clean up traffic vehicles."""
        for vehicle in self.vehicles:
            if vehicle is not None:
                try:
                    vehicle.destroy()
                except:
                    pass

        self.vehicles = []
        self.auto_vehicle_list = []
        self.actor_list = []

        if self.tm is not None:
            try:
                self.tm.shutdown()
            except:
                pass
            self.tm = None

    def get_vehicles(self):
        """Get list of traffic vehicles."""
        return self.vehicles

    def set_traffic_speed(self, speed):
        """Set speed for all traffic vehicles."""
        if self.tm is not None:
            for vehicle in self.vehicles:
                try:
                    self.tm.set_vehicle_speed(vehicle, speed)
                except:
                    pass

    def get_vehicle_positions(self):
        """Get current positions of all traffic vehicles."""
        positions = []
        for vehicle in self.vehicles:
            if vehicle is not None:
                try:
                    transform = vehicle.get_transform()
                    positions.append({
                        'x': transform.location.x,
                        'y': transform.location.y,
                        'z': transform.location.z,
                        'yaw': transform.rotation.yaw
                    })
                except:
                    pass
        return positions
