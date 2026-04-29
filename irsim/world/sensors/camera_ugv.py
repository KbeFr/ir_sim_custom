from math import cos, sin, tan, pi
from typing import TYPE_CHECKING

import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import numpy as np
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


class CameraUGV:
    """
    Simulates a UGV Camera sensor that detects objects within a triangular 
    view frustum (FOV cone).

    Args:
        state (np.ndarray): Initial state of the sensor.
        obj_id (int): ID of the associated object.
        max_range (float): Maximum detection distance of the camera.
        fov_angle (float): Field of view angle in radians.
        noise (bool): Whether noise is added to position measurements.
        std (float): Standard deviation for position noise.
        offset (list): Offset of the sensor from the object's position [x, y, theta].
        alpha (float): Transparency for plotting.
        **kwargs: Additional arguments.
            color (str): Color of the sensor's patch in plotting.
    """

    def __init__(
        self,
        state: np.ndarray | None = None,
        obj_id: int = 0,
        max_range: float = 5.0,
        fov_angle: float = pi / 4,
        noise: bool = False,
        std: float = 0.1,
        offset: list[float] | None = None,
        alpha: float = 0.2,
        **kwargs,
    ) -> None:
        
        if offset is None:
            offset = [0, 0, 0]
            
        self.sensor_type = "camera_ugv"
        self.obj_id = obj_id

        self.max_range = max_range
        self.fov_angle = fov_angle
        
        self.noise = noise
        self.std = std
        self.offset = np.c_[offset]
        self.alpha = alpha
        self.color = kwargs.get("color", "blue")
        
        self.detected_objects = []

        self.origin_state = np.zeros((3, 1))
        self._state = state if state is not None else self.origin_state
        
        self.init_geometry(self._state)

        # Parent object reference
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
        Initialize the UGV Camera's scanning geometry (A triangular FOV wedge).
        """
        # Half-width at the max range based on FOV
        w = tan(self.fov_angle / 2) * self.max_range

        # 1. Create the Triangular Sensing Wedge (Local coordinates)
        # Starts at [0,0] (the lens), ends at max_range spanning width w
        local_wedge_coords = np.array([
            [0.0, 0.0],
            [self.max_range, w],
            [self.max_range, -w]
        ])
        local_sensing_poly = Polygon(local_wedge_coords)

        # 2. Apply the Sensor Offset
        ox = self.offset[0, 0]
        oy = self.offset[1, 0]
        otheta = self.offset[2, 0] if self.offset.shape[0] > 2 else 0

        # Create offset transformation matrix
        rot_mat = np.array([
            [cos(otheta), -sin(otheta)],
            [sin(otheta), cos(otheta)]
        ])

        # Transform Sensing Poly by offset
        transformed_poly_coords = (rot_mat @ local_wedge_coords.T).T + np.array([ox, oy])
        self._original_sensing_poly = Polygon(transformed_poly_coords)

        # 3. Initialize world geometries based on starting state
        self.camera_origin = transform_point_with_state(self.offset, state)
        self._sensing_poly = geometry_transform(self._original_sensing_poly, state)
        self._geometry = self._sensing_poly  # Expose the sensing box as the primary geometry

    def step(self, state):
        """Update the Camera's state and process detections."""
        self._state = state

        # Update world states
        self.camera_origin = transform_point_with_state(self.offset, self._state)
        self._sensing_poly = geometry_transform(self._original_sensing_poly, self._state)
        self._geometry = self._sensing_poly

        # Run detection using the sensing wedge
        self.detected_objects = self._process_detection()

    


    def _process_detection(self):
        """Find objects intersecting the camera's FOV wedge."""
        object_tree = self._env_param.GeometryTree
        objects = self._env_param.objects
        geometries = [obj._geometry for obj in objects]

        detected = []
        if object_tree is None:
            return detected

        potential_geometries_index = object_tree.query(self._sensing_poly)

        for geom_index in potential_geometries_index:
            geo = geometries[geom_index]
            obj = objects[geom_index]

            if obj._id == self.obj_id or not obj._geometry_valid or obj.unobstructed:
                continue
            if obj.shape == "map":
                continue

            if self._sensing_poly.intersects(geo) or self._sensing_poly.contains(geo):
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
        """Convert world position to relative position relative to the camera origin."""
        cx = self.camera_origin[0, 0]
        cy = self.camera_origin[1, 0]
        ctheta = self.camera_origin[2, 0] if self.camera_origin.shape[0] > 2 else 0

        dx = world_pos[0, 0] - cx
        dy = world_pos[1, 0] - cy

        rel_x = dx * cos(-ctheta) - dy * sin(-ctheta)
        rel_y = dx * sin(-ctheta) + dy * cos(-ctheta)

        return np.array([[rel_x], [rel_y]])

    def get_detected_objects(self):
        return self.detected_objects

    def get_offset(self):
        return np.squeeze(self.offset).tolist()

    @property
    def state(self) -> np.ndarray:
        return self._state

    def plot(self, ax, state: np.ndarray | None = None, **kwargs):
        if state is None:
            state = self.state
        self._plot(ax, state, **kwargs)

    def _init_plot(self, ax, **kwargs):
        self._plot(ax, self.origin_state, **kwargs)

    def _plot(self, ax, state, **kwargs):
        """Plot the FOV Sensing Polygon."""
        
        # --- 2D Plotting ---
        if not isinstance(ax, Axes3D):
            # Plot Sensing Wedge
            poly_coords = np.array(self._original_sensing_poly.exterior.coords)
            self.camera_patch = mpatches.Polygon(
                poly_coords, closed=True, fill=True, color=self.color, alpha=self.alpha, zorder=2
            )
            ax.add_patch(self.camera_patch)

            # Apply Transformations
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

        # --- 3D Plotting Stub ---
        else:
            if state is not None and len(state) > 0:
                robot_x = state[0, 0]
                robot_y = state[1, 0]
                robot_theta = state[2, 0] if state.shape[0] > 2 else 0
            else:
                robot_x, robot_y, robot_theta = 0, 0, 0

            # 3D Polygon
            coords = np.array(self._original_sensing_poly.exterior.coords)
            world_coords = []
            for x, y in coords:
                wx = robot_x + x * cos(robot_theta) - y * sin(robot_theta)
                wy = robot_y + x * sin(robot_theta) + y * cos(robot_theta)
                world_coords.append([wx, wy, 0.0])

            self.camera_patch = Poly3DCollection(
                [world_coords], alpha=self.alpha, facecolors=self.color, edgecolors=self.color
            )
            ax.add_collection3d(self.camera_patch)
            
            self.plot_patch_list.append(self.camera_patch)

    def _step_plot(self):
        """Update the visualization transforms."""
        if not hasattr(self, "camera_patch"):
            return

        ax = self.camera_patch.axes
        if ax is None:
            return

        robot_x = self._state[0, 0]
        robot_y = self._state[1, 0]
        robot_theta = self._state[2, 0] if self._state.shape[0] > 2 else 0

        if isinstance(ax, Axes3D):
            # 3D Update logic (recalculating coords)
            coords = np.array(self._original_sensing_poly.exterior.coords)
            world_coords = []
            for x, y in coords:
                wx = robot_x + x * cos(robot_theta) - y * sin(robot_theta)
                wy = robot_y + x * sin(robot_theta) + y * cos(robot_theta)
                world_coords.append([wx, wy, 0.0])
            self.camera_patch.set_verts([world_coords])
            
        else:
            # 2D Update logic (using Affine2D transforms for performance)
            trans = (
                mtransforms.Affine2D()
                .rotate(robot_theta)
                .translate(robot_x, robot_y)
                + ax.transData
            )
            self.camera_patch.set_transform(trans)

    def step_plot(self):
        self._step_plot()

    def plot_clear(self):
        [patch.remove() for patch in self.plot_patch_list]
        self.plot_patch_list = []
        self.plot_line_list = []
        self.plot_text_list = []