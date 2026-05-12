import matplotlib
matplotlib.use("QtAgg")
import irsim

from overarchingTwin.overarching_twin import OverArchingTwin
from overarchingTwin.mission_planner import Mission, MissionType

from irsim.world.robots.uav_twin import UAVTwin
from irsim.world.robots.ugv_twin import UGVTwin
from local_planners.c3bf_qp import CollisionConeCBFController
from local_planners.cbf_qp import CBFQPController
from local_planners.local_planners import PurePursuitController
from mission_logger import MissionLogger

from simulation_gui import launch

# ---  Config 
CONTROLLER       = "cbf"          # "c3bf" | "cbf" | "pure_pursuit"
USE_GLOBAL_PLAN  = False            # False = UGV navigates to YAML goal directly
PERCEPTION_MODE  = "ugv"           # "all" | "uav" | "ugv" | "merged"
MAX_STEPS        = 800

# --- Environment
env = irsim.make("Experiment1/world_experiment1.yaml")

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


# --- Global logger 
mission_logger = MissionLogger()


# ---  Overarching Twin 
adt = OverArchingTwin(
    env             = env,
    uav             = uav_twins,
    ugv             = ugv_twins,
    mission_logger  = mission_logger,
    resolution      = 0.4,          # grid cell size [m]
)



# --- Mission definition 

# Each mission is independent of the YAML goal.
# The planner assigns them optimally across all available UGVs.

if USE_GLOBAL_PLAN:

    mission = Mission(
        mission_id=1,
        mission_type=MissionType.GOTO_WAYPOINT,
        goal_xy=(10,35),
        mission_posture="EXPLORE"
    )

    #Add a mission we want to complete to the overarching dt
    adt.add_mission(mission=mission)


launch(
    env             = env,
    adt             = adt,
    ugv_twins       = ugv_twins,
    controllers     = controllers,
    uav_twins       = uav_twins,
    max_steps       = MAX_STEPS,
    step_ms         = 100,           # 10 Hz — use the Speed slider to go faster
    perception_mode = PERCEPTION_MODE,
)


# End of simulation 

"""
mission_logger.draw_mission_costs(adt)
for ugv_id , metric_logger in adt._loggers.items():
    metric_logger.plot_figures()
"""

