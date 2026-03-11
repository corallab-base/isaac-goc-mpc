import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from isaacsim.util.debug_draw import _debug_draw

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_mul,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

from isaac_goc_mpc.ur5e.ur5e_robotiq_2f85 import UR5e_DIRECT_CONTROL_CFG, UR5e_CFG

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg


def setup_goc_mpc():
    import numpy as np
    from goc_mpc.goc_mpc import GraphOfConstraintsMPC
    from goc_mpc.splines import Block
    from goc_mpc._ext.goc_mpc import GraphOfConstraints
    from goc_mpc.simple_drake_env import SimpleDrakeGym

    env = SimpleDrakeGym(["free_body_0"], [], meshcat=None)

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0"], [], -10.0, 10.0)
    joint_agent_dim = graph.num_agents * graph.dim

    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    home_pose  = np.array([ 0.00, 0.30, 0.50, 0.0, 0.0, 1.0, 0.0])
    goal1_pose = np.array([-0.10, 0.30, 0.30, 0.0, 0.0, 1.0, 0.0])
    goal2_pose = np.array([ 0.10, 0.30, 0.30, 0.0, 0.0, 1.0, 0.0])

    A = np.eye(joint_agent_dim)
    graph.add_robots_linear_eq(0, A, home_pose)
    graph.add_robots_linear_eq(1, A, goal1_pose)
    graph.add_robots_linear_eq(2, A, goal2_pose)

    goc_mpc = GraphOfConstraintsMPC(
        graph, [Block.R(3), Block.SO3()],
        short_path_time_per_step=0.1,
        solve_for_waypoints_once=False,
        time_delta_cutoff=0.001,
        phi_tolerance=0.003,
        max_vel=1.0,
        max_acc=9.0,
    )
    return env, graph, goc_mpc


# Update robot states
def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
):
    """Update the robot states.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        robot: (Articulation) Robot articulation.
        ee_frame_idx: (int) End-effector frame index.
        arm_joint_ids: (list[int]) Arm joint indices.

    Returns:
        jacobian_b (torch.tensor): Jacobian in the body frame.
        mass_matrix (torch.tensor): Mass matrix.
        gravity (torch.tensor): Gravity vector.
        ee_pose_b (torch.tensor): End-effector pose in the body frame.
        ee_vel_b (torch.tensor): End-effector velocity in the body frame.
        root_pose_w (torch.tensor): Root pose in the world frame.
        ee_pose_w (torch.tensor): End-effector pose in the world frame.
        ee_force_b (torch.tensor): End-effector force in the body frame.
        joint_pos (torch.tensor): The joint positions.
        joint_vel (torch.tensor): The joint velocities.

    Raises:
        ValueError: Undefined target_type.
    """
    # obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]

    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
    root_vel_w = robot.data.root_vel_w  # Extract root velocity in the world frame
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Calculate the contact force
    ee_force_w = torch.zeros(scene.num_envs, 3, device=sim.device)
    sim_dt = sim.get_physics_dt()
    # Calculate the contact force by averaging over last four time steps (i.e., to smoothen) and
    # taking the max of three surfaces as only one should be the contact of interest
    ee_force_w = 0.0

    # This is a simplification, only for the sake of testing.
    ee_force_b = ee_force_w

    # Get joint positions and velocities
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    )


def main():

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground     = AssetBaseCfg(prim_path="/World/defaultGroundPlane",
                                  spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(prim_path="/World/Light",
                                  spawn=sim_utils.DomeLightCfg(intensity=3000.0,
                                                               color=(0.75, 0.75, 0.75)))
        robot: ArticulationCfg = UR5e_DIRECT_CONTROL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot")

    sim = SimulationContext(sim_utils.SimulationCfg(
        dt=0.01,
        # gravity=(0.0, 0.0, 0.0),
        device=args.device,
        save_logs_to_file=False
    ))
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.5])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()

    draw = _debug_draw.acquire_debug_draw_interface()

    robot = scene["robot"]

    # Obtain indices for the end-effector and arm joints
    ee_frame_name = "wrist_3_link"
    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # redundant?
    robot_entity_cfg = SceneEntityCfg("robot",
                                      joint_names=arm_joint_names,
                                      body_names=["wrist_3_link"])
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

    # Create the OSC
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["wrench_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=True,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=None,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
        contact_wrench_control_axes_task=[1, 1, 1, 1, 1, 1],
        nullspace_control="none",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    _, _, goc_mpc = setup_goc_mpc()
    goc_mpc.reset()

    sim_dt = sim.get_physics_dt()
    sim.step()
    scene.update(sim_dt)

    ee_pose_w0 = (robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]][0]
                  .detach().cpu().numpy())
    print("Initial EE pose (world):", ee_pose_w0)

    sim_steps_per_replan = 1
    prev_remaining     = None
    sim_tick           = 0
    MAX_STEPS          = 20000
    t                  = 0

    while simulation_app.is_running():

        # --- read robot state ------------------------------------------------
        jacobian   = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w  = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        joint_pos  = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_state_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0]]
        ee_com_acc_w = robot.data.body_com_acc_w[:, robot_entity_cfg.body_ids[0]]

        t += sim_dt
        x = torch.cat([ee_pose_w[:, :3], ee_pose_w[:, 3:7]], dim=-1)[0]\
                 .detach().cpu().numpy().astype(np.float64)
        x_dot = torch.cat([ee_state_w[:, 7:10], ee_state_w[:, 10:13]], dim=-1)[0]\
                     .detach().cpu().numpy().astype(np.float64)
        x_ddot = torch.cat([ee_com_acc_w[:, 0:3], ee_com_acc_w[:, 3:6]], dim=-1)[0]\
                     .detach().cpu().numpy().astype(np.float64)

        # --- GoC step (decimated) --------------------------------------------
        if sim_tick % sim_steps_per_replan == 0:
            try:
                _, vels, times = goc_mpc.step(t, x, x_dot, teleport=False)
            except RuntimeError as e:
                print("Error: ", e)
                # cached = goc_mpc.last_cycle_short_path
                # if cached is not None:
                #     points, vels, times = cached
                # else:
                #     points = np.expand_dims(x, 0)
                #     vels   = np.zeros((1, 6))
                pass

            agent_spline = goc_mpc.last_cycle_splines[0]
            begin_time = agent_spline.begin()
            end_time = agent_spline.end()

            _, _, acceleration_0 = agent_spline.eval(begin_time);

            # --- Cartesian acceleration command --------------------------------------
            a_cmd = torch.tensor(acceleration_0, dtype=jacobian.dtype, device=sim.device)

            # DEBUGGING SPLINE POINTS #########################################
            spline_times = np.linspace(begin_time, end_time, 100)
            agent_xi_l, _ = agent_spline.eval_multiple(spline_times)

            draw.clear_points()
            draw.draw_points(
                [(x[0], x[1], x[2]) for x in agent_xi_l[:, :3]],
                [(1, 0, 0, 1) for _ in agent_xi_l[:, :3]],
                [10 for _ in agent_xi_l[:, :3]]
            )
            # draw.draw_lines_spline([(x[0], x[1], x[2]) for x in agent_xi_l[:, :3]], (1, 0, 0, 1), 10, True)

            # Real velocity
            draw.clear_lines()
            # draw.draw_lines([
            #     (x[0], x[1], x[2])
            # ], [
            #     (x[0]+unit_x_dot[0], x[1]+unit_x_dot[1], x[2]+unit_x_dot[2]),
            # ], [(0, 1, 0, 1)], [10])

            unit_a_cmd = a_cmd / a_cmd.norm()

            draw.draw_lines([
                (x[0], x[1], x[2])
            ], [
                (x[0]+unit_a_cmd[0], x[1]+unit_a_cmd[1], x[2]+unit_a_cmd[2]),
            ], [(0, 0, 1, 1)], [10])

            # draw.draw_points([
            #     (x[0], x[1], x[2]),
            # ], [(1, 0, 0, 1)], [10])
            # unit_x_dot = x_dot / np.linalg.norm(x_dot)

            unit_x_ddot = x_ddot / np.linalg.norm(x_ddot)
            draw.draw_lines([
                (x[0], x[1], x[2])
            ], [
                (x[0]+unit_x_ddot[0], x[1]+unit_x_ddot[1], x[2]+unit_x_ddot[2]),
            ], [(0, 1, 0, 1)], [10])
            # DEBUGGING SPLINE POINTS #########################################

            # nodes_and_taus = list(zip(
            #     goc_mpc.timing_mpc.get_next_nodes(),
            #     goc_mpc.timing_mpc.get_next_taus()
            # ))

            # time_deltas_list = goc_mpc.timing_mpc.view_time_deltas_list()

            # rem = tuple(goc_mpc.remaining_phases)
            # if rem != prev_remaining:
            #     print("remaining_phases:", goc_mpc.remaining_phases,
            #           "completed_phases:", sorted(goc_mpc.completed_phases))
            #     prev_remaining = rem

        # get the updated states
        (
            jacobian_b,
            mass_matrix,
            gravity,
            ee_pose_b,
            ee_vel_b,
            root_pose_w,
            ee_pose_w,
            ee_force_b,
            joint_pos,
            joint_vel,
        ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)

        # 1. Compute the inverse of the joint-space mass matrix
        mass_matrix_inv = torch.inverse(mass_matrix)

        # 2. Compute the inverse of the operational space mass matrix
        # Shape of jacobian_b: [num_envs, 6, num_dof]
        # lambda_inv = J * M^-1 * J^T
        lambda_inv = jacobian_b @ mass_matrix_inv @ jacobian_b.transpose(1, 2)

        # 3. Solving for wrench from equation: lambda_inv * wrench = a_cmd
        # Use a small damping term to prevent singular matrix inversion
        eps = 1e-6
        wrench_cmd = torch.linalg.solve(lambda_inv + eps * torch.eye(6, device=sim.device), a_cmd.unsqueeze(-1)).squeeze(-1)

        # 5. Send to OSC
        command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
        command[:] = wrench_cmd

        task_frame_pose_b = torch.tensor([
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=command.dtype, device=sim.device)
        osc.set_command(command=command, current_task_frame_pose_b=task_frame_pose_b)

        # --- Convert end-effector wrench to joints velocity ----------------

        # compute the joint commands
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            # nullspace_joint_pos_target=joint_centers,
        )

        # if nodes_and_taus:
        #     next_node, next_tau = nodes_and_taus[0]

        #     print(f"[GoC t={t:.2f}] node={next_node}, tau={next_tau:.3f}, "
        #           f"remaining={goc_mpc.remaining_phases}\n"
        #           f"Current pos: [{x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}]\n"
        #           f"Current cmd: [{v_cmd[0]:.3f}, {v_cmd[1]:.3f}, {v_cmd[2]:.3f}]\n"
        #           f"Joint vels: [{joint_vel_des}]\n"
        #           f"Target waypoint 0: {goc_mpc.waypoint_mpc.view_waypoints()[0][:3]}")

        robot.set_joint_effort_target(joint_efforts * 0.9, joint_ids=arm_joint_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if len(goc_mpc.remaining_phases) == 0:
            print("GoC finished.")
            break

        sim_tick += 1
        if sim_tick >= MAX_STEPS:
            print("Hit MAX_STEPS.")
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
