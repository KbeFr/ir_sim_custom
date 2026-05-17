from irsim.world.robots.uav_twin import UAVTwin


class UAVFleetDT:
    """
    Aggregates obstacle detections and coverage footprints from all UAVs.
    """

    def __init__(self, uavs: list[UAVTwin] | UAVTwin) -> None:
        self._all_uavs = uavs if isinstance(uavs, list) else [uavs]

        self.first_time_view = True

        # Objects that hidden for fault injection
        self.hidden_objects = set()

    def get_uavs_view(self) -> list[dict]:
        if self.first_time_view:
            self.sensor_step()
            self.first_time_view = False

        obstacles = []
        for uav in self.uavs:
            obstacles.extend(uav.get_uav_view())

        return [obj for obj in obstacles if obj not in self.hidden_objects]

    """For real position return, not nessesary yet
    def get_uav_view(self) -> list[dict]:
        #Merged obstacle detections, deduped by id.
        seen: dict[int, dict] = {}
        for uav in self.uavs:
            for obj in uav.get_uav_view():
                seen[obj["id"]] = obj
        return list(seen.values())
    """

    def get_coverage_geometry(self) -> list:
        """Shapely geometries of all active UAV camera footprints."""
        geoms = []
        for uav in self.uavs:
            cam = next(
                (s for s in uav.sensors if s.sensor_type == "camera_uav"),
                None,
            )
            if cam is not None:
                geoms.append(cam._geometry)
        return geoms

    def sensor_step(self):
        for uav in self.uavs:
            uav.sensor_step()

    def add_uav(self, robot):
        if robot not in self._all_uavs:
            self._all_uavs.append(robot)

    def remove_uav(self, robot):
        if robot in self._all_uavs:
            self._all_uavs.remove(robot)
        else:
            print(f"[UAV Fleet] Robot: {robot.id} not in list, wanted to remove")


    @property
    def uavs(self) -> list[UAVTwin]:
        """
        Dynamically returns only the UAVs that are unobstructed/active.
        This prevents having to check visibility in every single loop.
        """
        return [u for u in self._all_uavs if not getattr(u, 'unobstructed', False)]
