import argparse
from isaaclab.app import AppLauncher


def move_ee():
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

    home_pose  = np.array([ 0.00, 0.30, 0.50, 1.0, 0.0, 0.0, 0.0])
    goal1_pose = np.array([-0.10, 0.30, 0.30, 1.0, 0.0, 0.0, 0.0])
    goal2_pose = np.array([ 0.10, 0.30, 0.30, 1.0, 0.0, 0.0, 0.0])

    A = np.eye(joint_agent_dim)
    graph.add_robots_linear_eq(0, A, home_pose)
    graph.add_robots_linear_eq(1, A, goal1_pose)
    graph.add_robots_linear_eq(2, A, goal2_pose)

    goc_mpc = GraphOfConstraintsMPC(
        graph, [Block.R(3), Block.SO3()],
        short_path_time_per_step=0.1,
        solve_for_waypoints_once=False,
        time_delta_cutoff=0.0,
        phi_tolerance=0.03,
    )
    return env, graph, goc_mpc


def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app

    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_mul, quat_inv

    from isaac_goc_mpc.ur5e.ur5e_robotiq_2f85 import UR5e_ROBOTIQ_2F_85_CFG
    from isaac_goc_mpc.controllers.differential_ik import (
        CartesianVelocityController, CartesianVelocityControllerCfg
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground     = AssetBaseCfg(prim_path="/World/defaultGroundPlane",
                                  spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(prim_path="/World/Light",
                                  spawn=sim_utils.DomeLightCfg(intensity=3000.0,
                                                               color=(0.75, 0.75, 0.75)))
        robot: ArticulationCfg = UR5e_ROBOTIQ_2F_85_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device,
                                                    save_logs_to_file=False))
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.5])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()

    robot = scene["robot"]

    ARM_JOINT_NAMES = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    robot_entity_cfg = SceneEntityCfg("robot",
                                      joint_names=ARM_JOINT_NAMES,
                                      body_names=["wrist_3_link"])
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

    vel_controller = CartesianVelocityController(
        CartesianVelocityControllerCfg(device=sim.device, damping=0.05)
    )

    # GoC passes a node only when the timing MPC's planned time-to-node (tau0)
    # drops below the GoC step size (0.3 s).  If the robot tracks poorly, tau0
    # stays large and phases are never completed.  Higher gains → tighter
    # tracking → tau0 shrinks on time → phases complete.
    K_POS = 10.0  # linear  (1/s)
    K_ORI =  5.0  # angular (1/s)

    env, graph, goc_mpc = move_ee()
    goc_mpc.reset()

    sim_dt = sim.get_physics_dt()
    sim.step()
    scene.update(sim_dt)

    ee_pose_w0 = (robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]][0]
                  .detach().cpu().numpy())
    print("Initial EE pose (world):", ee_pose_w0)

    goc_dt               = goc_mpc.short_path_time_per_step   # 0.1 s
    goc_step             = 3
    sim_steps_per_replan = max(1, int(round(goc_step * goc_dt / sim_dt)))  # 30

    goc_k            = 0
    current_target_w   = None
    current_target_vel = None
    prev_remaining     = None
    sim_tick           = 0
    MAX_STEPS          = 20000

    while simulation_app.is_running():

        # --- read robot state ------------------------------------------------
        jacobian   = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w  = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        joint_pos  = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_state_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0]]

        x = torch.cat([ee_pose_w[:, :3], ee_pose_w[:, 3:7]], dim=-1)[0]\
                    .detach().cpu().numpy().astype(np.float64)
        x_dot = torch.cat([ee_state_w[:, 7:10], ee_state_w[:, 10:13]], dim=-1)[0]\
                        .detach().cpu().numpy().astype(np.float64)

        # --- GoC step (decimated) --------------------------------------------
        if sim_tick % sim_steps_per_replan == 0 or current_target_w is None:
            t_goc = goc_k * goc_dt
            try:
                points, vels, times = goc_mpc.step(t_goc, x, x_dot, teleport=False)
            except RuntimeError as e:
                print(f"[GoC] {e}")
                cached = goc_mpc.last_cycle_short_path
                if cached is not None:
                    points, vels, times = cached
                else:
                    points = np.expand_dims(x, 0)
                    vels   = np.zeros((1, 6))

            idx = min(goc_step, points.shape[0] - 1)
            current_target_w   = points[idx, :7].copy()
            current_target_vel = (vels[idx, :6].copy()
                                  if vels is not None and vels.shape[1] >= 6
                                  else np.zeros(6))
            goc_k += goc_step

            pos_err = np.linalg.norm(current_target_w[:3] - x[:3])
            print(f"[GoC t={t_goc:.2f}]  target={current_target_w[:3].round(3)}"
                  f"  ee={x[:3].round(3)}  err={pos_err:.4f}m")

            rem = tuple(goc_mpc.remaining_phases)
            if rem != prev_remaining:
                print("remaining_phases:", goc_mpc.remaining_phases,
                      "completed_phases:", sorted(goc_mpc.completed_phases))
                prev_remaining = rem

        # --- Cartesian velocity command --------------------------------------
        tgt  = torch.tensor(current_target_w, device=sim.device,
                            dtype=ee_pose_w.dtype).unsqueeze(0)
        v_ff = torch.tensor(current_target_vel, device=sim.device,
                            dtype=ee_pose_w.dtype).unsqueeze(0)

        e_pos = tgt[:, :3] - ee_pose_w[:, :3]
        q_err = quat_mul(tgt[:, 3:7], quat_inv(ee_pose_w[:, 3:7]))
        e_ang = 2.0 * torch.sign(q_err[:, :1]) * q_err[:, 1:4]

        v_cmd = torch.cat([K_POS * e_pos + v_ff[:, :3],
                           K_ORI * e_ang + v_ff[:, 3:]], dim=-1)

        # --- Integrate joint velocity to position target ---------------------
        joint_vel_des = vel_controller.compute(v_cmd, jacobian)
        joint_pos_des = joint_pos + joint_vel_des * sim_dt
        robot.set_joint_position_target(joint_pos_des,
                                        joint_ids=robot_entity_cfg.joint_ids)

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
