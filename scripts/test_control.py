import argparse
import math
import time

from isaaclab.app import AppLauncher

def move_ee():
    import numpy as np

    from goc_mpc.goc_mpc import GraphOfConstraintsMPC
    from goc_mpc.splines import Block
    from goc_mpc._ext.goc_mpc import GraphOfConstraints
    from goc_mpc.simple_drake_env import SimpleDrakeGym

    env = SimpleDrakeGym(["free_body_0"], [], meshcat=None)

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0"], [],
                               state_lower_bound, state_upper_bound)
    agent_dim = graph.dim;
    joint_agent_dim = graph.num_agents * graph.dim

    # Construct graph
    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    # Identity quaternion for stability (wxyz)
    home_pose = np.array([0.00, 0.30, 0.50, 1.0, 0.0, 0.0, 0.0])
    goal1_pose = np.array([-0.10, 0.30, 0.30, 1.0, 0.0, 0.0, 0.0])
    goal2_pose = np.array([ 0.10, 0.30, 0.30, 1.0, 0.0, 0.0, 0.0])

    A = np.eye(joint_agent_dim)  # pins all 7 dims

    graph.add_robots_linear_eq(0, A, home_pose)
    graph.add_robots_linear_eq(1, A, goal1_pose)
    graph.add_robots_linear_eq(2, A, goal2_pose)

    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(
        graph, spline_spec,
        short_path_time_per_step=0.1,
        solve_for_waypoints_once=False,
        time_delta_cutoff=0.0,
        phi_tolerance=0.03,
    )
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

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device, save_logs_to_file=False))
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

    # GoC stuff...
    env, graph, goc_mpc = move_ee()

    # Reset
    diff_ik_controller.reset()
    goc_mpc.reset()

    sim_dt = sim.get_physics_dt()

    # Take 1 sim step so robot.data buffers are valid
    sim.step()
    scene.update(sim_dt)
    
    # Debug initial position
    ee_pose_w0 = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]][0].detach().cpu().numpy()
    print("Initial EE pose (world):", ee_pose_w0)
    
    # ----------------------------
    # Planner-rate decimation setup
    # ----------------------------
    MAX_STEPS = 5000
    sim_tick = 0

    goc_dt = goc_mpc.short_path_time_per_step   # e.g. 0.1
    goc_step = 3                                # match the example
    replan_period = goc_step * goc_dt           # e.g. 0.3 seconds
    sim_steps_per_replan = max(1, int(round(replan_period / sim_dt)))

    goc_k = 0
    current_target_w = None

    prev_remaining = None
    
    # simulation loop
    while simulation_app.is_running():
    
        # 1) Read robot state (torch, world frame)
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
    
        ee_state_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0]]  # (N,13)
        ee_linvel_w = ee_state_w[:, 7:10]
        ee_angvel_w = ee_state_w[:, 10:13]
    
        # 2) Build GoC inputs (numpy, single-agent)
        x = torch.cat([ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]], dim=-1)[0].detach().cpu().numpy()
        x_dot = torch.cat([ee_linvel_w, ee_angvel_w], dim=-1)[0].detach().cpu().numpy()
    
        # (Recommended) Drake/GoC usually prefers float64
        x = x.astype(np.float64)
        x_dot = x_dot.astype(np.float64)
    
        # 3) Call GoC only every N sim steps
        if (sim_tick % sim_steps_per_replan) == 0 or current_target_w is None:
            t_goc = goc_k * goc_dt  # absolute GoC time like k*dt in the example
    
            try:
                points, vels, times = goc_mpc.step(t_goc, x, x_dot, teleport=False)
            except RuntimeError as e:
                print(e)
                points, vels, times = goc_mpc.last_cycle_short_path
                if points is None:
                    points = np.expand_dims(x, axis=0)
    
            # sample a point consistent with our planning stride
            idx = min(goc_step, points.shape[0] - 1)
            current_target_w = points[idx, :7].copy()
    
            # advance GoC time in big jumps like the example (k += step)
            goc_k += goc_step
    
            # Debug: print phase changes (only when we replan, so it won't spam)
            rem = tuple(goc_mpc.remaining_phases)
            if rem != prev_remaining:
                print("remaining_phases:", goc_mpc.remaining_phases,
                      "completed_phases:", sorted(list(goc_mpc.completed_phases)))
                prev_remaining = rem
    
        # Use held target every sim tick
        target_w = current_target_w
    
        # 4) Convert target world pose -> base frame for DiffIK
        target_pose_w = torch.tensor(target_w, device=sim.device, dtype=root_pose_w.dtype).unsqueeze(0)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pose_w[:, 0:3], target_pose_w[:, 3:7],
        )
        ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
        diff_ik_controller.set_command(ik_command)
    
        # 5) Current EE pose in base frame (needed for compute)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )
    
        # 6) Compute joint targets + apply
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
    
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    
        # Termination conditions
        if len(goc_mpc.remaining_phases) == 0:
            print("GoC finished. Stopping.")
            break
    
        sim_tick += 1
        if sim_tick >= MAX_STEPS:
            print("Hit MAX_STEPS. Stopping.")
            break
    
    simulation_app.close()

if __name__ == "__main__":
    main()

