"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR5E_CFG`: The UR5E arm without a gripper.
* :obj:`UR5E_ROBOTIQ_2F_85_CFG`: The UR5E arm with Robotiq 2F-85 gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from importlib.resources import files

# from isaac_goc_mpc.actuators.actuator_vel_pd import IdealVelocityPDActuatorCfg
import isaac_goc_mpc

##
# Configuration
##

UR5e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(files(isaac_goc_mpc) / "ur5e" / "ur5e_robotiq_2f85.usd"),
        variants={"Gripper": "None"},
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0000,
            "shoulder_lift_joint": -2.2000,
            "elbow_joint": 1.9000,
            "wrist_1_joint": -1.3830,
            "wrist_2_joint": -1.5700,
            "wrist_3_joint": 0.0000,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64101615,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
    },
    articulation_root_prim_path="/root_joint",
)

"""Configuration of UR5e arm with zero stiffness for velocity control."""

UR5e_VELOCITY_CONTROL_CFG = UR5e_CFG.copy()
UR5e_VELOCITY_CONTROL_CFG.actuators={
    "shoulder": ImplicitActuatorCfg(
        joint_names_expr=["shoulder_.*"],
        stiffness=0.0, # 1320.0,
        damping=72.6636085,
        friction=0.0,
        armature=0.0,
    ),
    "elbow": ImplicitActuatorCfg(
        joint_names_expr=["elbow_joint"],
        stiffness=0.0, # 600.0,
        damping=34.64101615,
        friction=0.0,
        armature=0.0,
    ),
    "wrist": ImplicitActuatorCfg(
        joint_names_expr=["wrist_.*"],
        stiffness=0.0, # 216.0,
        damping=29.39387691,
        friction=0.0,
        armature=0.0,
    ),
}

"""Configuration of UR5e arm with zero stiffness and dampening for direct effort control."""

UR5e_DIRECT_CONTROL_CFG = UR5e_CFG.copy()
UR5e_DIRECT_CONTROL_CFG.actuators={
    "shoulder": ImplicitActuatorCfg(
        joint_names_expr=["shoulder_.*"],
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        armature=0.0,
    ),
    "elbow": ImplicitActuatorCfg(
        joint_names_expr=["elbow_joint"],
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        armature=0.0,
    ),
    "wrist": ImplicitActuatorCfg(
        joint_names_expr=["wrist_.*"],
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        armature=0.0,
    ),
}


# UR5e_VELOCITY_CONTROL_CFG.actuators={
#     "shoulder": IdealVelocityPDActuatorCfg(
#         joint_names_expr=["shoulder_.*"],
#         stiffness=0.0, # 1320.0,
#         damping=72.6636085,
#         friction=0.0,
#         armature=0.0,
#     ),
#     "elbow": IdealVelocityPDActuatorCfg(
#         joint_names_expr=["elbow_joint"],
#         stiffness=0.0, # 600.0,
#         damping=34.64101615,
#         friction=0.0,
#         armature=0.0,
#     ),
#     "wrist": IdealVelocityPDActuatorCfg(
#         joint_names_expr=["wrist_.*"],
#         stiffness=0.0, # 216.0,
#         damping=29.39387691,
#         friction=0.0,
#         armature=0.0,
#     ),
# }

"""Configuration of UR5e arm with Robotiq_2f_85 gripper."""

UR5e_ROBOTIQ_2F_85_CFG = UR5e_CFG.copy()
"""Configuration of UR5e arm with Robotiq_2f_85 gripper."""
UR5e_ROBOTIQ_2F_85_CFG.spawn.variants = {"Gripper": "Robotiq_2f_85"}
UR5e_ROBOTIQ_2F_85_CFG.spawn.rigid_props.disable_gravity = True
UR5e_ROBOTIQ_2F_85_CFG.init_state.joint_pos["finger_joint"] = 0.0
UR5e_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_inner_finger_joint"] = 0.0
UR5e_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_inner_finger_knuckle_joint"] = 0.0
UR5e_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_outer_.*_joint"] = 0.0
# the major actuator joint for gripper
UR5e_ROBOTIQ_2F_85_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
)
# enable the gripper to grasp in a parallel manner
UR5e_ROBOTIQ_2F_85_CFG.actuators["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.2,
    damping=0.001,
    friction=0.0,
    armature=0.0,
)
# set PD to zero for passive joints in close-loop gripper
UR5e_ROBOTIQ_2F_85_CFG.actuators["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)
