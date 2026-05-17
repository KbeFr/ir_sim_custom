import matplotlib

matplotlib.use("QtAgg")

from local_controllers.c3bf_qp import CollisionConeCBFController
from local_controllers.cbf_qp import CBFQPController
from local_controllers.local_planners import PurePursuitController
from loggers.mission_logger import MissionLogger
from overarching_twin.overarching_twin import OverArchingTwin
from simulation_gui import launch

import irsim
from irsim.world.robots.uav_twin import UAVTwin
from irsim.world.robots.ugv_twin import UGVTwin

# ---  Config
CONTROLLER       = "cbf"          # "c3bf" | "cbf" | "pure_pursuit"
PERCEPTION_MODE  = "ALL"           # "all" | "uav" | "ugv" | "merged"

# --- Environment
env = irsim.make("project_kobe_frateur/experiments/Experiment1/world_experiment1.yaml")

# Collect UAV and UGV twins from the robot list.
uav_twins = [r for r in env.robot_list if isinstance(r, UAVTwin)]
ugv_twins = [r for r in env.robot_list if isinstance(r, UGVTwin)]


# --- Controllers (one per UGV)
controllers = {}
for ugv in ugv_twins:
    if CONTROLLER == "c3bf":
        controllers[ugv.id] = CollisionConeCBFController(
            robot_type    = ugv.kinematics,
            safety_margin = 0.2,
            goal_gain     = 0.8,
        )
    elif CONTROLLER == "cbf":
        controllers[ugv.id] = CBFQPController(
            robot_type=ugv_twins[0].kinematics,
            safety_margin=0.2,
            cbf_alpha=1.0,
            goal_gain=0.8,
        )
    elif CONTROLLER == "pure_pursuit":
        controllers[ugv.id] = PurePursuitController()


# --- Mission logger
mission_logger = MissionLogger()


# ---  Overarching Twin
adt = OverArchingTwin(
    env             = env,
    uav             = uav_twins,
    ugv             = ugv_twins,
    perception_mode = PERCEPTION_MODE,
    mission_logger  = mission_logger,
    resolution      = 0.4,          # grid cell size [m]
)

# -- Launch the gui and simulation loop
launch(
    env             = env,
    adt             = adt,
    controllers     = controllers,
    step_ms         = 100,           # 10 Hz
    perception_mode = PERCEPTION_MODE,
)

