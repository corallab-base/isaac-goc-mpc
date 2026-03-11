import argparse
import math
import time

from isaaclab.app import AppLauncher

def move_ee():
    env = SimpleDrakeGym(["free_body_0"], [], meshcat=None)

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0"], [],
                               state_lower_bound, state_upper_bound)
    agent_dim = graph.dim;


    # Construct graph
    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    goal_position_1 = np.array([-0.10, 0.3, 0.3, 0.0, 0.0, 1.0, 0.0])
    phi0 = graph.add_robots_linear_eq(0, 
                                      np.eye(joint_agent_dim), 
                                      goal_position_1)

    goal_position_2 = np.array([0.10, 0.3, 0.3, 0.0, 0.0, 1.0, 0.0])
    phi1 = graph.add_robots_linear_eq(1, 
                                      np.eye(joint_agent_dim), 
                                      goal_position_2)
   
    # MARK: I wonder if this needs to match the initial robot position?
    home_position_1 = np.array([0.00, 0.3, 0.5, 0.0, 0.0, 1.0, 0.0])


    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, 
                                    short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc 

def main():

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    # --- TODO: Add optional args here ---

    # ---
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app


    # Isaac imports AFTER app is created
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.utils.math import subtract_frame_transforms

    # Robot cfg
    from isaac_goc_mpc.ur5e.ur5e_robotiq_2f85 import UR5e_ROBOTIQ_2F_85_CFG 

    # GoC-MPC + Drake imports
    from goc_mpc.goc_mpc import GraphOfConstraintsMPC
    from goc_mpc.splines import Block
    from goc_mpc._ext.goc_mpc import GraphOfConstraints

    from pydrake.systems.framework import DiagramBuilder
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.multibody.parsing import Parser
    from importlib.resources import files
    import goc_mpc as goc_pkg

    # Scene configuration
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
                              spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, 
                                         color=(0.75, 0.75, 0.75)),
        )
        robot: ArticulationCfg = UR5e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.5])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))

    sim.reset()
    scene.reset()

    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", 
                                              use_relative_mode=False, 
                                              ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, 
                                                  num_envs=scene.num_envs, 
                                                  device=sim.device)
    
    # Hard-coded EE + arm joints
    EE_BODY_NAME = "wrist_3_link"

    ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    robot_entity_cfg = SceneEntityCfg("robot", 
                                      joint_names=ARM_JOINT_NAMES, 
                                      body_names=[EE_BODY_NAME])
    robot_entity_cfg.resolve(scene)

    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

    # TODO: GoC stuff... 
    env, graph, goc_mpc = move_ee()
    dt = goc_mpc.short_path_time_per_step

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():

        # TODO: GoC updates?
        obs, _ = env.reset()
        goc_mpc.reset()

        # TODO: Clean this up
        # obtain quantities from simulation
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # compute the joint commands
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        
        # TODO: Pass the ...????
        xi_h, _, _ = goc_mpc.step(sim_dt, x, x_dot)
                
        # TODO: apply actions (I assume need to use xi_h somehow?)
        robot.set_joint_position_target(joint_pos_des, 
                                        joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7] 

    app.close()

if __name__ == "__main__":
    main()

