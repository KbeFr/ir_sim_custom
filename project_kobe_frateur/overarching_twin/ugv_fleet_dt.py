from irsim.world.robots.ugv_twin import UGVTwin


class UGVFleetDT:
    """
    Aggregates obstacle detections and management from all UGVs.
    """

    def __init__(self, ugvs: list[UGVTwin] | UGVTwin) -> None:
        self._all_ugvs = ugvs if isinstance(ugvs, list) else [ugvs]

        self.first_time_view = True
        # Objects that hidden for fault injection
        self.hidden_objects = set()

    def get_ugvs_view(self):
        if self.first_time_view:
            self.sensor_step()
            self.first_time_view = False

        objects = []
        for ugv in self.ugvs:
            objects.extend(ugv.get_ugv_view())
        return objects

    def sensor_step(self):
        for ugv in self._all_ugvs:
            ugv.sensor_step()

    def get_ugv(self, ugv_id: str) -> UGVTwin | None:
        for u in self._all_ugvs:
            if u.id == ugv_id:
                return u
        return None

    def add_ugv(self, robot):
        if robot not in self._all_ugvs:
            self._all_ugvs.append(robot)

    def remove_ugv(self, robot):
        if robot in self._all_ugvs:
            self._all_ugvs.remove(robot)
        else:
            print(f"[OverArchingTwin] Robot: {robot.id} not in list, wanted to remove")

    @property
    def ugvs(self) -> list[UGVTwin]:
        """
        Dynamically returns only the UGVs that are unobstructed/active.
        This prevents having to check visibility in every single loop.
        """
        return [u for u in self._all_ugvs if not getattr(u, 'unobstructed', False)]

    @property
    def all_ugvs(self) -> list[UGVTwin]:
        """
        Dynamically returns only the UGVs that are unobstructed/active.
        This prevents having to check visibility in every single loop.
        """
        return self._all_ugvs
