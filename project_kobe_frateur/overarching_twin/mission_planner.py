"""
mission_planner.py  —  Mission Definition and Assignment Framework
==================================================================
Implements the mission layer of the HDT architecture.

"""

from __future__ import annotations

import math
import time

import numpy as np

from project_kobe_frateur.overarching_twin.mission import (
    POSTURE_WEIGHTS,
    Mission,
    MissionType,
)

from ..loggers.mission_logger import MissionLogger


class MissionPlanner:
    """
    Handles mission assignment and path planning for the Overarching Twin.
    """

    def __init__(
        self,
        astar_planner,
        grid_map,
        sim_time_fn,
        uav_world_map_fn,
        mission_logger : MissionLogger,
        safety_reserve: float = 15.0,
    ) -> None:
        self._planner       = astar_planner
        self._grid_map      = grid_map
        self._sim_time      = sim_time_fn       # callable: () → float
        self._uav_world_map = uav_world_map_fn  # callable: () → dict
        self._safety_reserve = safety_reserve

        self.mission_logger = mission_logger


    def assign_and_plan(
        self,
        missions:  list[Mission],
        ugv_list:  list,
    ) -> dict[str, np.ndarray]:
        """
        Assign pending missions to available UGVs and plan paths.

        Returns
        -------
        dict {ugv_id: path_array (2, N)}  for each newly assigned pair.
        Already-active missions are not re-assigned unless forced.
        """

        pending   = [m for m in missions if m.status == "pending"]
        available = [u for u in ugv_list
                     if not self._ugv_busy(u, missions)]


        if not pending or not available:
            return {}

        # Cost matrix
        n_ugv = len(available)
        n_mis = len(pending)
        C = np.full((n_ugv, n_mis), fill_value=1e9)

        # For each mission pending
        for j, mission in enumerate(pending):

            # First get the goal based on mission type
            goal = self._resolve_goal(mission)

            #Then we go over all available ugvs and check cost path
            for i, ugv in enumerate(available):

                if goal is None:
                    continue   # goal not yet available (TIME_GATED, target hidden)

                start_xy = ugv.state
                t0 = time.perf_counter()
                path, cost = self._plan(ugv, start_xy, goal, POSTURE_WEIGHTS[mission.mission_posture])
                print(
                    f"[MissionPlanner] Path planned, "
                    f"Lenght : {len(path[0])}   "
                    f"{(time.perf_counter() - t0) * 1000:.0f} ms"
                )

                if self._check_battery(ugv, path):
                    C[i, j] = cost
                    mission.last_cost = cost
                else :
                    print(f"[MissionPlanner] Robot {self._ugv_id(ugv)} Cant complete mission because battery to low ")
                    C[i, j] = 1e9   # infeasible


                self.mission_logger.update_per_mission_log(mission.mission_id, self._ugv_id(ugv) , path , cost , C[i, j])



        # Hungarian assignment based on cost matrix
        try:
            from scipy.optimize import linear_sum_assignment
            row_idx, col_idx = linear_sum_assignment(C)
        except ImportError:
            # Fallback: greedy nearest assignment
            row_idx, col_idx = self._greedy_assign(C)

        result: dict[str, np.ndarray] = {}

        # Finalize assignment
        for i, j in zip(row_idx, col_idx, strict=False):
            if C[i, j] >= 1e8:
                continue   # no feasible assignment

            ugv     = available[i]
            mission = pending[j]

            if goal is None:
                continue

            ugv_id              = self._ugv_id(ugv)
            mission.assigned_ugv = ugv_id
            mission.status       = "active"

            ugv.assigned_mission = mission

            result[ugv_id]       = path

            self.mission_logger.update_assignment_log(
                mission.mission_id,
                ugv_id,
                self._sim_time(),
                cost,
                "initial_assignment")

            print(
                f"[MissionPlanner] Successfully linked mission to ground robot"
                f"{ugv_id} → {mission.mission_id} "
                f"cost={cost:.2f}  goal={goal}"
            )

        return result


    def replan(
        self,
        ugv,
        mission: Mission,
        weights: tuple,
        reason:  str = "triggered",
    ) -> np.ndarray | None:
        """
        Replan a specific UGV/mission pair.  Called by the Overarching Twin
        when a dynamic obstacle enters the path or battery drops.
        """
        ugv_pos = self._ugv_xy(ugv)
        goal    = self._resolve_goal(mission, ugv_pos)
        if goal is None:
            return None

        path, cost = self._plan(ugv, ugv_pos, goal, weights)
        if self._check_battery(ugv, path):
            self.mission_logger.update_assignment_log(
                        mission.mission_id,
                        self._ugv_id(ugv),
                        self._sim_time(),
                        cost,
                        reason=reason)
            return path

        print(f"[MissionPlanner] Replan failed for {self._ugv_id(ugv)}: Battery to low")
        return None

    def posture_for_battery(self, battery_pct: float) -> str:
        """
        Return the recommended PBPA posture based on battery state.
        """
        if battery_pct > 60:
            return "EXPLORE"
        if battery_pct > 30:
            return "CONSERVE"
        return "URGENT"

    @property
    def assignment_log(self) -> list[dict]:
        return self._assignment_log

    # ── Private helpers ───────────────────────────────────────────────────────

    def _plan(
        self,
        ugv,
        start_xy: tuple,
        goal_xy:  tuple,
        weights:  tuple,
    ) -> tuple[np.ndarray, float]:
        """Run A* and return (path_array, total_cost)."""
        start = np.array([[start_xy[0]], [start_xy[1]]])
        goal  = np.array([[goal_xy[0]],  [goal_xy[1]]])


        path = self._planner.planning(
            start_pose     = start,
            goal_pose      = goal,
            #weights        = weights,
            #global_grid_map   = self._grid_map,
            #ugv            = ugv,
            show_animation = False,
        )
        return path, 1

    def _check_battery(self, ugv, path: np.ndarray) -> None:
        """Raise BatteryConstraintError if path energy exceeds budget."""
        battery = getattr(ugv, 'battery_status', 100.0)
        mass    = getattr(ugv, 'mass', 1.0)


        budget  = battery - self._safety_reserve


        energy = predict_path_energy(path, robot_mass=mass)

        print(f"[MissionPlanner] : Battery Check ; Battery : {battery} ; Safety : {self._safety_reserve} ; Energy Predicted {energy} ")

        if energy > budget:
            print(f"Path needs {energy:.1f}% battery, only {budget:.1f}% available.")
            return False
        return True

    def _resolve_goal(
        self,
        mission:  Mission,
    ) -> tuple[float, float] | None:
        """Extract the current (x, y) goal from any mission type."""
        if mission.mission_type == MissionType.GOTO_WAYPOINT:
            return mission.goal_xy

        if mission.mission_type == MissionType.TIME_GATED_GOTO:
            if self._sim_time() >= mission.unlock_time:
                return mission.goal_xy
            return None   # not yet unlocked

        if mission.mission_type == MissionType.TRACK_TARGET:
            world_map = self._uav_world_map()
            entry     = world_map.get(mission.target_id)
            if entry:
                return (entry["estimated_pos"][0], entry["estimated_pos"][1])
            return None   # target not in UAV coverage


        return None

    def _ugv_xy(self, ugv) -> tuple[float, float]:
        s = ugv.state
        return (float(s[0, 0]), float(s[1, 0]))

    def _ugv_id(self, ugv) -> str:
        return getattr(ugv, 'id', str(id(ugv)))

    def _ugv_busy(self, ugv, missions: list[Mission]) -> bool:
        """True if this UGV already has an active mission."""
        ugv_id = self._ugv_id(ugv)
        return any(
            m.assigned_ugv == ugv_id and m.status == "active"
            for m in missions
        )

    def _greedy_assign(self, C: np.ndarray) -> tuple:
        """
        Fallback greedy assignment when scipy is unavailable.
        Each UGV takes the cheapest available mission.
        """
        n_ugv, _n_mis = C.shape
        assigned_mis: set[int] = set()
        rows, cols = [], []
        for i in range(n_ugv):
            best_j = int(np.argmin(C[i]))
            if C[i, best_j] < 1e8 and best_j not in assigned_mis:
                rows.append(i)
                cols.append(best_j)
                assigned_mis.add(best_j)
        return rows, cols


def predict_path_energy(
    path_xy:    np.ndarray,
    robot_mass: float,
    v_avg:      float = 0.4,
    Ka:         float = 0.5,
    Ku:         float = 0.05,
) -> float:
    """
    Estimate the battery percentage consumed to traverse path_xy.

    Parameters
    ----------
    path_xy     : (2, N) array of world-metre waypoints.
    robot_mass  : total mass including payload [kg].
    v_avg       : average speed [m/s].
    Ka          : ancillary power constant [W].
    Ku          : rolling friction coefficient.

    Returns
    -------
    float : estimated energy in normalised battery-% units.
            Calibrate the scale factor to your specific robot.
    """



    g = 9.81
    xs, ys = path_xy[0], path_xy[1]
    total = 0.0
    for i in range(len(xs) - 1):
        d  = math.hypot(xs[i+1] - xs[i], ys[i+1] - ys[i])
        dt = d / max(v_avg, 1e-6)
        # Maneuvering energy [J]
        E_move = 2.0 * Ku * robot_mass * g * d
        # Ancillary energy [J]
        E_aux  = Ka * dt
        total += E_move + E_aux

    # Convert joules → battery % (scale factor; calibrate to hardware)
    JOULES_PER_PERCENT = 15
    return total / JOULES_PER_PERCENT
