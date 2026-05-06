import numpy as np
from irsim.util.random import rng
from irsim.world.object_base import ObjectBase

UAV_ALTITUDE = 3.0 
UAV_BATTERY = 100.0
UAV_BATTERY_DRAIN = 0.05  # Adjusted for smoother draining per step

class UAVTwin(ObjectBase):
    
    def __init__(
        self,
        shape: dict | None = None,
        kinematics: dict | None = None,
        state: list | None = None,
        velocity: list | None = None,
        goal: list | None = None,
        role: str = "robot",
        color: str = "k",
        static: bool = False,
        vel_min: list | None = None,
        vel_max: list | None = None,
        acce: list | None = None,
        angle_range: list | None = None,
        behavior: dict | None = None,
        group_behavior: dict | None = None,
        goal_threshold: float = 0.1,
        sensors: dict | None = None,
        arrive_mode: str = "position",
        description: str | None = None,
        group: int = 0,
        group_name: str | None = None,
        state_dim: int | None = None,
        vel_dim: int | None = None,
        unobstructed: bool = False,
        fov: float | None = None,
        fov_radius: float | None = None,
        name: str | None = None,
        **kwargs,
    ) -> None:        
        
        # 1. Pass all parameters up to the ObjectBase to initialize kinematics & geometry
        super().__init__(
            shape=shape, kinematics=kinematics, state=state, velocity=velocity, goal=goal,
            role=role, color=color, static=static, vel_min=vel_min, vel_max=vel_max,
            acce=acce, angle_range=angle_range, behavior=behavior, group_behavior=group_behavior,
            goal_threshold=goal_threshold, sensors=sensors, arrive_mode=arrive_mode,
            description=description, group=group, group_name=group_name, state_dim=state_dim,
            vel_dim=vel_dim, unobstructed=unobstructed, fov=fov, fov_radius=fov_radius,
            name=name, **kwargs
        )
        
        # 2. Initialize Twin-specific properties
        self.altitude = UAV_ALTITUDE
        self.battery_status = UAV_BATTERY
    

    def get_uav_view(self):
        """
        Returns the data collected by the UÄV's onboard camera (if attached).
        """
        # Search the sensor array for the CameraUAV built earlier
        camera = next((sensor for sensor in self.sensors if sensor.sensor_type == "camera_uav"), None)
        
        if camera:
            return camera.get_detected_objects()
        else:
            self.logger.warning(f"UGV {self.id} requested view but has no Camera sensor attached.")
            return []


    """ This Is for the uncertainty with higher altitude, will implement later 
    def get_uav_view(self):
        #Returns the position of each object in the environment.
        #Applies positional uncertainty (noise) that scales linearly with the UAV's altitude.
        detected = []
        
        # Noise scales with altitude (e.g. 3.0m alt * 0.05 = 0.15m standard deviation)
        altitude_noise_std = self.altitude * 0.05 
        
        for obj in self.external_objects:
            if obj.shape == "map" or not obj._geometry_valid:
                continue
                
            # Extract true world position
            world_pos = obj.state[0:2]
            
            # Apply altitude-based uncertainty to the perceived position
            estimated_pos = world_pos + rng.normal(0, altitude_noise_std, (2, 1))
            
            # Extract shape information for ROS mapping / path planning
            shape_info = {"name": obj.shape}
            if obj.shape == "circle":
                shape_info["radius"] = getattr(obj, "radius", 0.0)
            elif obj.shape in ["rectangle", "polygon"]:
                if hasattr(obj, "length"):
                    shape_info["length"] = obj.length
                    shape_info["width"] = getattr(obj, "width", 0.0)
                if hasattr(obj, "vertices"):
                    shape_info["vertices"] = np.array(obj.vertices).tolist()
                    
            detected.append({
                "id": obj.id,
                "estimated_pos": estimated_pos.flatten().tolist(),
                "true_pos": world_pos.flatten().tolist(),  # Useful for benchmarking
                "shape": shape_info
            })
            
        return detected
    """
    def step(self, velocity: np.ndarray | None = None, sensor_step: bool = True, **kwargs):
        """
        Override the step function to include battery dynamics.
        """
        # Drain battery if active
        if not self.static and not self.stop_flag:
            if self.battery_status > 0:
                self.battery_status -= UAV_BATTERY_DRAIN
            else:
                self.battery_status = 0
                self.stop_flag = True
                self.logger.warning(f"UAV {self.id} has run out of battery and stopped!")

        # Execute standard physics/kinematics step
        return super().step(velocity=velocity, sensor_step=sensor_step, **kwargs)