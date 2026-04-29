from irsim.world.robots.uav_twin import UAVTwin
from irsim.world.robots.ugv_twin import UGVTwin



class UAVFleetDT:
    """
    Aggregates obstacle detections and coverage footprints from all UAVs.
    """

    def __init__(self, uavs: list[UAVTwin] | UAVTwin) -> None:
        self.uavs = uavs if isinstance(uavs, list) else [uavs]

    def get_uav_view(self) -> list[dict]:
        """Merged obstacle detections, deduped by id."""
        seen: dict[int, dict] = {}
        for uav in self.uavs:
            for obj in uav.get_uav_view():
                seen[obj["id"]] = obj
        return list(seen.values())

    def get_coverage_geometry(self) -> list:
        """Shapely geometries of all active UAV camera footprints."""
        geoms = []
        for uav in self.uavs:
            cam = next(
                (s for s in uav.sensors if s.sensor_type == "camera"),
                None,
            )
            if cam is not None:
                geoms.append(cam._geometry)
        return geoms

