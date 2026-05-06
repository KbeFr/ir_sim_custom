import numpy as np
from irsim.world.object_base import ObjectBase

UGV_MASS = 10
UGV_BATTERY = 100.0
UGV_BATTERY_DRAIN = 0.1

AVG_SPEED = 1
ANC_DRAIN = 0.05

class UGVTwin(ObjectBase):
    
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
        
        # Pass all parameters up to ObjectBase
        super().__init__(
            shape=shape, kinematics=kinematics, state=state, velocity=velocity, goal=goal,
            role=role, color=color, static=static, vel_min=vel_min, vel_max=vel_max,
            acce=acce, angle_range=angle_range, behavior=behavior, group_behavior=group_behavior,
            goal_threshold=goal_threshold, sensors=sensors, arrive_mode=arrive_mode,
            description=description, group=group, group_name=group_name, state_dim=state_dim,
            vel_dim=vel_dim, unobstructed=unobstructed, fov=fov, fov_radius=fov_radius,
            name=name, **kwargs
        )

        
        # UGV specific logic
        self.battery_status = UGV_BATTERY
        self.mass = UGV_MASS 
        self.avg_speed = AVG_SPEED
        self.ancillary_drain = ANC_DRAIN #[w]


        self.assigned_mission = None


    def get_ugv_view(self):
        """
        Returns the data collected by the UGV's onboard camera (if attached).
        """
        # Search the sensor array for the CameraUGV built earlier
        camera = next((sensor for sensor in self.sensors if sensor.sensor_type == "camera_ugv"), None)
        
        if camera:
            return camera.get_detected_objects()
        else:
            self.logger.warning(f"UGV {self.id} requested view but has no Camera sensor attached.")
            return []

    def step(self, velocity: np.ndarray | None = None, sensor_step: bool = True, **kwargs):
        """
        Override the step function to include ground-vehicle battery dynamics.
        """
        # Drain battery if active
        if not self.static and not self.stop_flag:
            if self.battery_status > 0:
                self.battery_status -= UGV_BATTERY_DRAIN
            else:
                self.battery_status = 0
                self.stop_flag = True
                self.logger.warning(f"UGV {self.id} has run out of battery and stopped!")


        """
        velocity (np.ndarray, optional): Desired velocity for this step.
                If None, the object will use its behavior system to generate velocity.
                The shape and meaning depend on the kinematics model:

                - Differential: [linear_velocity, angular_velocity]
                - Omnidirectional: [velocity_x, velocity_y]
                - Ackermann: [linear_velocity, steering_angle]
        
        """
        # Execute standard physics/kinematics step
        return super().step(velocity=velocity, sensor_step=sensor_step, **kwargs)