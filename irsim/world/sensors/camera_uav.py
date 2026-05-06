from math import cos, sin
from typing import TYPE_CHECKING

import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import numpy as np
import shapely
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon

from irsim.util.random import rng
from irsim.util.util import (
    geometry_transform,
    transform_point_with_state,
)

if TYPE_CHECKING:
    from irsim.world.object_base import ObjectBase


class CameraUAV:
    """
    Simulates a Camera sensor that detects objects within a rectangular bounding box.

    Args:
        state (np.ndarray): Initial state of the sensor.
        obj_id (int): ID of the associated object.
        length (float): Length of the rectangular sensing area (forward/backward).
        width (float): Width of the rectangular sensing area (left/right).
        noise (bool): Whether noise is added to position measurements.
        std (float): Standard deviation for position noise.
        offset (list): Offset of the sensor from the object's position [x, y, theta].
        alpha (float): Transparency for plotting.
        **kwargs: Additional arguments.
            color (str): Color of the sensor's patch in plotting.

    Attr:
        - sensor_type (str): Type of sensor ("camera").
        - length (float): Detection box length in meters.
        - width (float): Detection box width in meters.
        - offset (np.ndarray): Offset of the sensor relative to the object's position.
        - camera_origin (np.ndarray): Origin position of the Camera sensor in world coordinates.
        - detected_objects (list): List of dicts detailing the position of objects detected.
    """

    def __init__(
        self,
        state: np.ndarray | None = None,
        obj_id: int = 0,
        length: float = 4.0,
        width: float = 3.0,
        noise: bool = False,
        std: float = 0.1,
        offset: list[float] | None = None,
        alpha: float = 0.2,
        **kwargs,
    ) -> None:
        """
        Initialize the Camera sensor.
        """
        if offset is None:
            offset = [0, 0, 0]
            
        self.sensor_type = "camera_uav"
        self.obj_id = obj_id

        self.length = length
        self.width = width
        self.noise = noise
        self.std = std
        self.offset = np.c_[offset]
        self.alpha = alpha

        self.color = kwargs.get("color", "blue")
        
        self.detected_objects = []

        self.origin_state = np.zeros((3, 1))
        self._state = state if state is not None else self.origin_state
        self.init_geometry(self._state)

        # Parent object reference (set by ObjectBase or SensorFactory)
        self.parent: ObjectBase | None = None

        self.plot_patch_list = []
        self.plot_line_list = []
        self.plot_text_list = []

    @property
    def _env_param(self):
        """Access env_param via parent's env instance if available."""
        if self.parent is not None and self.parent._env is not None:
            return self.parent._env._env_param
        from irsim.config import env_param

        return env_param

    def init_geometry(self, state):
        """
        Initialize the Camera's scanning geometry (a rectangular polygon).

        Args:
            state (np.ndarray): Current state of the sensor.
        """


        # Half dimensions
        l = self.length / 2
        w = self.width / 2

        # Extract offset
        ox = self.offset[0, 0] 
        oy = self.offset[1, 0]
        otheta = self.offset[2, 0] if self.offset.shape[0] > 2 else 0

        # Local unrotated box corners centered at 0,0
        local_coords = np.array([
            [l, w],
            [-l, w],
            [-l, -w],
            [l, -w]
        ])

        # Rotate corners by the offset theta, and translate by offset x, y
        rot_mat = np.array([
            [cos(otheta), -sin(otheta)],
            [sin(otheta), cos(otheta)]
        ])

        transformed_coords = (rot_mat @ local_coords.T).T + np.array([ox, oy])
        
        # The base geometry relative to the robot center
        self._original_geometry = Polygon(transformed_coords)

        # The geometry relative to the world
        self.camera_origin = transform_point_with_state(self.offset, state)
        self._geometry = geometry_transform(self._original_geometry, state)
        self._init_geometry = self._geometry

    def step(self, state):
        """
        Update the Camera's state and process intersections with environment objects.

        Args:
            state (np.ndarray): New state of the sensor parent/robot.
        """
        self._state = state

        # Update world states
        self.camera_origin = transform_point_with_state(self.offset, self._state)
        self._geometry = geometry_transform(self._original_geometry, self._state)

        # Run detection
        self.detected_objects = self._process_detection()

    def _process_detection(self):
        """
        Find objects intersecting the camera's rectangular box.

        Returns:
            list[dict]: A list of detected objects and their positions.
        """
        object_tree = self._env_param.GeometryTree
        objects = self._env_param.objects
        geometries = [obj._geometry for obj in objects]

        detected = []
        
        if object_tree is None:
            return detected
        
        # Find potential objects near the camera geometry
        potential_geometries_index = object_tree.query(self._geometry)

        for geom_index in potential_geometries_index:
            geo = geometries[geom_index]
            obj = objects[geom_index]

            # Ignore itself, invalid objects, or unobstructed items
            if obj._id == self.obj_id or not obj._geometry_valid or obj.unobstructed:
                continue

            # We usually ignore the map lines for camera positional detection unless specifically desired
            if obj.shape == "map":
                continue

            # Check if the object intersects or is fully contained within the camera view
            if self._geometry.intersects(geo) or self._geometry.contains(geo):
                detected.append(obj)  
        return detected          
        """
                
                world_pos = obj.state[0:2]

                if self.noise:
                    world_pos = world_pos + rng.normal(0, self.std, (2, 1))

                rel_pos = self._get_relative_pos(world_pos)

                detected.append({
                    "id": obj._id,
                    "world_pos": world_pos.flatten().tolist(),
                    "relative_pos": rel_pos.flatten().tolist()
                })

        return detected
        """

    def _get_relative_pos(self, world_pos):
        """
        Convert world position to relative position relative to the camera origin.
        """
        cx = self.camera_origin[0, 0]
        cy = self.camera_origin[1, 0]
        ctheta = self.camera_origin[2, 0] if self.camera_origin.shape[0] > 2 else 0

        dx = world_pos[0, 0] - cx
        dy = world_pos[1, 0] - cy

        # Inverse rotation
        rel_x = dx * cos(-ctheta) - dy * sin(-ctheta)
        rel_y = dx * sin(-ctheta) + dy * cos(-ctheta)

        return np.array([[rel_x], [rel_y]])

    def get_detected_objects(self):
        """
        Get the detected objects and their positions.

        Returns:
            list[dict]: Detected objects data containing ids and positions.
        """
        return self.detected_objects

    def get_offset(self):
        """
        Get the sensor's offset.

        Returns:
            list: Offset as a list.
        """
        return np.squeeze(self.offset).tolist()

    def plot(self, ax, state: np.ndarray | None = None, **kwargs):
        """Plot the Camera patch on a given axis."""
        if state is None:
            state = self.state

        self._plot(ax, state, **kwargs)

    def _init_plot(self, ax, **kwargs):
        """Initialize the plot for the Camera."""
        self._plot(ax, self.origin_state, **kwargs)

    @property
    def state(self) -> np.ndarray:
        return self._state

    def _plot(self, ax, state, **kwargs):
        """
        Plot the Camera's rectangular box using the specified state for positioning.
        """
        if isinstance(ax, Axes3D):
            if state is not None and len(state) > 0:
                robot_x = state[0, 0]
                robot_y = state[1, 0]
                robot_theta = state[2, 0] if state.shape[0] > 2 else 0
            else:
                robot_x, robot_y, robot_theta = 0, 0, 0

            coords = np.array(self._original_geometry.exterior.coords)
            world_coords = []
            
            for x, y in coords:
                wx = robot_x + x * cos(robot_theta) - y * sin(robot_theta)
                wy = robot_y + x * sin(robot_theta) + y * cos(robot_theta)
                world_coords.append([wx, wy, 0.0])

            self.camera_patch = Poly3DCollection(
                [world_coords], alpha=self.alpha, facecolors=self.color, edgecolors='k'
            )
            ax.add_collection3d(self.camera_patch)
            
        else:
            # 2D Polygons plotted in local coordinates then offset by Transform
            coords = np.array(self._original_geometry.exterior.coords)
            self.camera_patch = mpatches.Polygon(
                coords, closed=True, fill=True, color=self.color, alpha=self.alpha, zorder=2
            )
            ax.add_patch(self.camera_patch)

            if state is not None and len(state) > 0:
                robot_x = state[0, 0]
                robot_y = state[1, 0]
                robot_theta = state[2, 0] if state.shape[0] > 2 else 0

                trans = (
                    mtransforms.Affine2D()
                    .rotate(robot_theta)
                    .translate(robot_x, robot_y)
                    + ax.transData
                )
                self.camera_patch.set_transform(trans)

        self.plot_patch_list.append(self.camera_patch)

    def _step_plot(self):
        """
        Update the Camera visualization using matplotlib transforms based on current state.
        """
        if not hasattr(self, "camera_patch"):
            return

        ax = self.camera_patch.axes
        if ax is None:
            return

        if isinstance(ax, Axes3D):
            robot_x = self._state[0, 0]
            robot_y = self._state[1, 0]
            robot_theta = self._state[2, 0] if self._state.shape[0] > 2 else 0

            coords = np.array(self._original_geometry.exterior.coords)
            world_coords = []
            for x, y in coords:
                wx = robot_x + x * cos(robot_theta) - y * sin(robot_theta)
                wy = robot_y + x * sin(robot_theta) + y * cos(robot_theta)
                world_coords.append([wx, wy, 0.0])
                
            self.camera_patch.set_verts([world_coords])
        else:
            robot_x = self._state[0, 0]
            robot_y = self._state[1, 0]
            robot_theta = self._state[2, 0] if self._state.shape[0] > 2 else 0

            trans = (
                mtransforms.Affine2D()
                .rotate(robot_theta)
                .translate(robot_x, robot_y)
                + ax.transData
            )
            self.camera_patch.set_transform(trans)

    def step_plot(self):
        """Public method to update the visualization."""
        self._step_plot()

    def plot_clear(self):
        """Clear the plot elements from the axis."""
        [patch.remove() for patch in self.plot_patch_list]
        [line.pop(0).remove() for line in self.plot_line_list]
        [text.remove() for text in self.plot_text_list]

        self.plot_patch_list = []
        self.plot_line_list = []
        self.plot_text_list = []