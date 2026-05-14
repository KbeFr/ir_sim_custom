from __future__ import annotations

import math
from enum import Enum

import numpy as np

from irsim.env.env_base import EnvBase
from irsim.world.robots.uav_twin import UAVTwin
from irsim.world.robots.ugv_twin import UGVTwin

from ..loggers.metrics_logger import MetricsLogger
from ..loggers.mission_logger import MissionLogger
from ..path_planners.a_star import AStarPlanner
from .grid_map import GlobalGridMap
from .mission import DEFAULT_POSTURE, POSTURE_WEIGHTS, Mission
from .mission_planner import MissionPlanner
from .uav_fleet_dt import UAVFleetDT


class PerceptionMode(Enum):
    ALL = 1
    UGV = 2
    UAV = 3
    MERGED = 4

    @classmethod
    def get_names(cls):
        return [member.name for member in cls]


class OverArchingTwin:
    """
    World DT + Mission DT.

    Parameters
    ----------
    env             ir-sim environment.
    uav             UAVTwin or list thereof.
    ugv             UGVTwin or list thereof.
    resolution      Grid cell size [m].  0.3-0.5 m for a 40 m world.
    open_map_viewer Open a separate map-viewer window (set False for headless).
    """

    def __init__(
            self,
            env: EnvBase,
            uav: list[UAVTwin] | UAVTwin,
            ugv: list[UGVTwin] | UGVTwin,
            mission_logger: MissionLogger,
            resolution: float = 0.4,
    ) -> None:
        self.env = env

        # --- World dimensions
        self._W = float(env._world.width)
        self._H = float(env._world.height)
        self._ox = float(env._world.offset[0])
        self._oy = float(env._world.offset[1])
        self._world_specs = (self._W, self._H, self._ox, self._oy)
        self.resolution = resolution

        # --- Agents
        self.uav_fleet = UAVFleetDT(uav)
        self._all_ugvs: list[UGVTwin] = ugv if isinstance(ugv, list) else [ugv]

        # --- Define innit obstacles view
        self.perception_mode: PerceptionMode = PerceptionMode.ALL
        self.perceived_obstacles = env.obstacle_list

        # ---  global grid map + cost map
        self.grid_map = GlobalGridMap(
            world_specs=self._world_specs,
            obstacle_list=self.perceived_obstacles,
            resolution=resolution,
        )

        # --- A* planner
        self._astar = AStarPlanner(env_map=self.grid_map.env_map)

        # --- mission planner
        self.missions: list[Mission] = []
        self._posture: str = DEFAULT_POSTURE
        self._sim_step: int = 0

        self._dt: float = 0.1

        self.mission_planner = MissionPlanner(
            astar_planner=self._astar,
            grid_map=self.grid_map,
            sim_time_fn=lambda: self._sim_step * self._dt,
            uav_world_map_fn=lambda: {
                obj["id"]: obj for obj in self.uav_fleet.get_uav_view()
            },
            mission_logger=mission_logger
        )

        self._loggers: dict[str, MetricsLogger] = {}
        self._setup_robot_loggers()

        # --- Cached path for dynamic-obstacle replan check
        self._active_paths: dict[str, np.ndarray] = {}
        self._replan_log: list[dict] = []

    def _setup_robot_loggers(self):
        # --- per-UGV metrics loggers
        self._loggers: dict[str, MetricsLogger] = {
            self._ugv_id(u): MetricsLogger(
                ugv_id=self._ugv_id(u),
                dt=self._dt,
                label=self._ugv_id(u),
            )
            for u in self.ugvs
        }


    def set_perception_mode(self, mode: str):
        mode = PerceptionMode[mode]

        print(f"[OVERACHING] set_perception_mode called, previous: {self.perception_mode}, new : {mode} ")

        if mode == self.perception_mode:
            pass
        elif mode == PerceptionMode.ALL:
            self.perceived_obstacles = self.env.obstacle_list
            self.perception_mode = mode
        elif mode == PerceptionMode.UAV:
            self.perceived_obstacles = self.get_uavs_view()
            self.perception_mode = mode
        elif mode == PerceptionMode.UGV:
            self.perceived_obstacles = self.get_ugvs_view()
            self.perception_mode = mode
        elif mode == PerceptionMode.MERGED:
            self.perceived_obstacles = self.get_merged_view()
            self.perception_mode = mode

        # update gridmap
        self.grid_map.update_perception(self.perceived_obstacles)

    def get_merged_view(self):
        object = []
        object.extend(self.get_uavs_view())
        object.extend(self.get_ugvs_view())
        return object

    def get_ugvs_view(self):
        # Step sensors if sim didnt run before, otherwise no detections present
        if self._sim_step == 0:
            self.ugvs_sensor_step()

        objects = []
        for ugv in self.ugvs:
            objects.extend(ugv.get_ugv_view())
        return objects

    def get_uavs_view(self):
        # Step sensors if sim didnt run before, otherwise no detections present
        if self._sim_step == 0:
            self.uav_fleet.sensor_step()
        return self.uav_fleet.get_uavs_view()

    def ugvs_sensor_step(self):
        for ugv in self.ugvs:
            ugv.sensor_step()

    def add_perceived_obstacle(self, obs):
        if obs not in self.perceived_obstacles:
            self.perceived_obstacles.append(obs)
        self.grid_map.update_perception(self.perceived_obstacles)

    def remove_perceived_obstacles(self, obs):
        if obs in self.perceived_obstacles:
            self.perceived_obstacles.remove(obs)
        self.grid_map.update_perception(self.perceived_obstacles)




    def reset(self):
        self._sim_step = 0
        self._setup_robot_loggers()

    """
    MAIN STEP
    """

    def step(self) -> None:

        self._sim_step += 1

        # Update UAV coverage
        geoms = self.uav_fleet.get_coverage_geometry()
        self.grid_map.update_coverage(geoms)

        # Assign best ugv for mission and weights
        new_paths = self.mission_planner.assign_and_plan(
            missions=self.missions,
            ugv_list=self.ugvs,
        )

        # Push waypoints to ugvs
        for ugv_id, path in new_paths.items():
            self._active_paths[ugv_id] = path
            ugv = self._get_ugv(ugv_id)
            if ugv is not None:
                self._send_path_to_ugv(ugv, path)

        # --- Dynamic obstacle replan check (every 10 steps)
        if self._sim_step % 10 == 0:
            # self._check_dynamic_replan()
            pass

        # --- Record metrics
        for ugv in self.ugvs:
            self._record_step(ugv)

    def add_mission(self, mission: Mission) -> None:
        """Register a mission for assignment in the next step()."""
        self.missions.append(mission)
        print(f"[OverArchingTwin] Mission added: {mission.mission_id} ({mission.mission_type.name})")

    # ══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _send_path_to_ugv(self, ugv: UGVTwin, path: np.ndarray) -> None:
        """Convert (2,N) path array to waypoints and call ugv.set_goal()."""
        xs, ys = path[0][::-1], path[1][::-1]  # reverse: goal→start → start→goal
        waypoints = [[float(x), float(y), 0.0]
                     for x, y in zip(xs[1:], ys[1:], strict=False)]
        ugv.set_goal(waypoints)

    def _check_dynamic_replan(self) -> None:
        """Trigger replanning if any dynamic obstacle enters an active path."""
        for ugv in self.ugvs:
            uid = self._ugv_id(ugv)
            path = self._active_paths.get(uid)
            if path is None:
                continue
            pts = list(zip(path[0], path[1], strict=False))
            for obs in self.env.obstacle_list:
                if not self._is_dynamic(obs):
                    continue
                ox = float(obs.state[0, 0])
                oy = float(obs.state[1, 0])
                r = getattr(obs, 'radius', 0.5) + 0.5
                for px, py in pts:
                    if math.hypot(px - ox, py - oy) < r:
                        print(
                            f"[OverArchingTwin] Dynamic obs {obs.id} on "
                            f"{uid} path — replanning."
                        )
                        self._replan_log.append({
                            "step": self._sim_step,
                            "ugv_id": uid,
                            "reason": "dynamic_obstacle",
                        })
                        # Find active mission and replan
                        mission = next(
                            (m for m in self.missions
                             if m.assigned_ugv == uid and m.status == "active"),
                            None,
                        )
                        if mission:
                            new_path = self.mission_planner.replan(
                                ugv, mission,
                                POSTURE_WEIGHTS[self._posture],
                                reason="dynamic_obstacle",
                            )
                            if new_path is not None:
                                self._active_paths[uid] = new_path
                                self._send_path_to_ugv(ugv, new_path)
                        return

    def _record_step(self, ugv):
        """Record a step of the ugv in logger"""

        uid = self._ugv_id(ugv)

        # Get grid index from real ugv pos
        gx, gy = self.grid_map.world_to_cell(ugv.state[0, 0], ugv.state[1, 0])

        # Get UGV coverage
        in_cov = (gx, gy) in self.grid_map.coverage

        # Record this data in the ugv specific logger
        if uid in self._loggers:
            self._loggers[uid].record(
                ugv=ugv,
                in_coverage=in_cov, )

    def _is_dynamic(self, obs) -> bool:
        if getattr(obs, 'static', True):
            return False
        if getattr(obs, 'behavior', None) is not None:
            return True
        vel_max = getattr(obs, 'vel_max', None)

        return bool(vel_max is not None and np.any(np.abs(np.array(vel_max).flatten()) > 1e-06))

    def _ugv_id(self, ugv) -> str:
        return getattr(ugv, 'id', str(id(ugv)))

    def _get_ugv(self, ugv_id: str) -> UGVTwin | None:
        for u in self.ugvs:
            if self._ugv_id(u) == ugv_id:
                return u
        return None

    @property
    def ugvs(self) -> list[UGVTwin]:
        """
        Dynamically returns only the UGVs that are unobstructed/active.
        This prevents having to check visibility in every single loop.
        """
        return [u for u in self._all_ugvs if not getattr(u, 'unobstructed', False)]
