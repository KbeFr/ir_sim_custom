# custom_world.py
import irsim

from overarchingTwin.overarching_twin import OverArchingTwin
from overarchingTwin.mission_planner import Mission, MissionType

from irsim.world.robots.uav_twin import UAVTwin
from irsim.world.robots.ugv_twin import UGVTwin
from local_planners.c3bf_qp import CollisionConeCBFController
from local_planners.cbf_qp import CBFQPController
from local_planners.local_planners import PurePursuitController
from mission_logger import MissionLogger

# ---  Config 
CONTROLLER       = "c3bf"          # "c3bf" | "cbf" | "pure_pursuit"
USE_GLOBAL_PLAN  = False            # False = UGV navigates to YAML goal directly
PERCEPTION_MODE  = "ugv"           # "all" | "uav" | "ugv" | "merged"
MAX_STEPS        = 800

# --- Environment
env = irsim.make("custom_world.yaml")

# Collect UAV and UGV twins from the robot list. 
uav_twins = [r for r in env.robot_list if isinstance(r, UAVTwin)]
ugv_twins = [r for r in env.robot_list if isinstance(r, UGVTwin)]

print(f"UAVs: {len(uav_twins)}  UGVs: {len(ugv_twins)}")
assert uav_twins, "No UAVTwin found — check"
assert ugv_twins, "No UGVTwin found — check"


# --- Controllers (one per UGV) 
controllers = {}
for ugv in ugv_twins:
    if CONTROLLER == "c3bf":
        controllers[ugv.id] = CollisionConeCBFController(
            robot_type    = ugv.kinematics,
            safety_margin = 0.05,
            goal_gain     = 0.8,
        )
    elif CONTROLLER == "cbf":
        controllers[ugv.id] = CBFQPController(
            robot_type=ugv_twins[0].kinematics,
            safety_margin=0.15,
            cbf_alpha=2.0,
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


# Simulation loop 
for step_i in range(MAX_STEPS):

    # OverArchingTwin step
    adt.step()

    # Per-UGV local control
    actions = []
    ids     = []

    for ugv in ugv_twins:

        #Dont advance the ugv if no mission is assigned to it
        if USE_GLOBAL_PLAN and ugv.assigned_mission is None:
            continue

        # Get obstacles in UGV sensor view for controller
        obstacles = ugv.get_ugv_view()

        # Get safe velocity command from the local controller
        ctrl    = controllers[ugv.id]
        action  = ctrl.get_action(ugv, obstacles)

        actions.append(action)
        ids.append(ugv.id)

    # Advance physics
    env.step(action=actions, action_id=ids)
    env.render(0.01)

    if env.done():
        print(f"[Sim] Done at step {step_i + 1}")
        break

# End of simulation 


mission_logger.draw_mission_costs(adt)

for ugv_id , metric_logger in adt._loggers.items():
    metric_logger.plot_figures()


env.end(3)