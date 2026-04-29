# Modifications to ir-sim for HDT Integration

This document outlines all the modifications made to the base [ir-sim](https://github.com/hanruihua/ir-sim) project required to run our custom Heterogeneous Digital Twin (HDT) architecture. 

**Base Release:** `ir-sim-v2.9.2`

To support UAV and UGV digital twins without breaking the original `ir-sim` functionality, we injected custom child classes, added new sensor definitions, and updated the environment configuration to recognize them.

---

## 1. Robot Digital Twins
First, two wrapper classes inheriting from `ObjectBase` were created to enable extra functionality and custom specifications for the UAVs and UGVs.

* **Location:** Found in `project_kobe_frateur/custom_components/robots`
* **Action:** Move these files to the `irsim/world/robots` directory so they are importable.

### Updating the `ObjectFactory`
To load these digital twins via the YAML file, update `irsim/world/object_factory.py`.

**Add imports at the top:**
```python
from irsim.world.robots.ugv_twin import UGVTwin
from irsim.world.robots.uav_twin import UAVTwin
```

**Modify line ~156:**
```python
if obj_type == "robot":
    object_list.append(self.create_robot("robot" , **obj_dict))
elif obj_type == "obstacle":
    object_list.append(self.create_obstacle(**obj_dict))
elif obj_type == "uav":
    object_list.append(self.create_robot("uav", **obj_dict))
elif obj_type == "ugv":
    object_list.append(self.create_robot("ugv", **obj_dict))
```

**Modify line ~206:**
```python
if type == "uav":
    print("UAV Created")
    return UAVTwin(kinematics=kinematics, role="robot", **kwargs)
elif type == "ugv":
    print("UGV Created")
    return UGVTwin(kinematics=kinematics, role="robot", **kwargs)
else:
    print("ROBOT Created")
    return ObjectBase(kinematics=kinematics, role="robot", **kwargs)
```

---

## 2. Custom Sensors
We added two custom sensors representing the monocular cameras of the UGV (a pie-shaped forward detection zone) and the UAV (a top-down rectangular coverage window). 

* **Location:** Found in `project_kobe_frateur/custom_components/sensors`
* **Action:** Move these files to the `irsim/world/sensors` directory.

### Updating the `SensorFactory`
To allow these sensors to be defined in the YAML config, update `irsim/world/sensors/sensor_factory.py`.

**Add imports at the top:**
```python
from irsim.world.sensors.camera_uav import CameraUAV 
from irsim.world.sensors.camera_ugv import CameraUGV 
```

**Modify line ~28:**
```python
if sensor_type == "lidar2d":
    return Lidar2D(state, obj_id, **kwargs)
elif sensor_type == "camerauav":
    return CameraUAV(state, obj_id, **kwargs)
elif sensor_type == "cameraugv":
    return CameraUGV(state, obj_id, **kwargs)
```

---

## 3. Environment Configuration
To aggregate the new UAV and UGV YAML blocks into the environment, small additions are required in `irsim/env/env_config.py`.

**Modify line ~55:**
```python
        self._kwargs_parse: dict[str, Any] = {
            "world": {},
            "gui": {},
            "robot": None,
            "uav": None,    # <- added
            "ugv": None,    # <- added
            "obstacle": None,
        }        
```

**Modify lines ~112 and ~183:**
Change the standard robot loading logic to the following block to seamlessly aggregate standard robots, UAVs, and UGVs:
```python
        robot_collection = []
        group_start = 0
        
        # Aggregate any definitions under robot, uav, and ugv keys
        for r_type in ["robot", "uav", "ugv"]:
            parsed_data = self.parse.get(r_type)
            if parsed_data is not None:
                new_bots = self.object_factory.create_from_parse(
                    parsed_data, r_type, group_start_index=group_start
                )
                robot_collection.extend(new_bots)
                group_start = max((obj.group for obj in robot_collection), default=-1) + 1
```