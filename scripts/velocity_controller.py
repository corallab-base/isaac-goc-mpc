import argparse
from isaaclab.app import AppLauncher


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
        time_delta_cutoff=0.1,
        phi_tolerance=0.05,
        max_vel=0.05,
        max_acc=7.0,
    )
    return env, graph, goc_mpc


def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app

    import numpy as np
    import torch

    from isaacsim.util.debug_draw import _debug_draw

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_mul, quat_inv

    from isaac_goc_mpc.ur5e.ur5e_robotiq_2f85 import UR5e_VELOCITY_CONTROL_CFG
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
        robot: ArticulationCfg = UR5e_VELOCITY_CONTROL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot")

    sim = SimulationContext(sim_utils.SimulationCfg(
        dt=0.01,
        gravity=(0.0, 0.0, 0.0),
        device=args.device,
        save_logs_to_file=False
    ))
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.5])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()

    draw = _debug_draw.acquire_debug_draw_interface()

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

        t += sim_dt
        x = torch.cat([ee_pose_w[:, :3], ee_pose_w[:, 3:7]], dim=-1)[0]\
                 .detach().cpu().numpy().astype(np.float64)
        x_dot = torch.cat([ee_state_w[:, 7:10], ee_state_w[:, 10:13]], dim=-1)[0]\
                     .detach().cpu().numpy().astype(np.float64)

        # draw.draw_points([
        #     (x[0], x[1], x[2]),
        # ], [(1, 0, 0, 1)], [10])

        unit_x_dot = x_dot / np.linalg.norm(x_dot)

        # Real velocity
        draw.clear_lines()
        # draw.draw_lines([
        #     (x[0], x[1], x[2])
        # ], [
        #     (x[0]+unit_x_dot[0], x[1]+unit_x_dot[1], x[2]+unit_x_dot[2]),
        # ], [(0, 1, 0, 1)], [10])

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

            # --- Cartesian target velocity
            # command---------------------------------- Use the next velocity in
            # the short horizon as a target, which is an approximation to
            # using the acceleration of the spline at time 0 as a command.
            v_cmd = torch.tensor(vels[1], dtype=jacobian.dtype, device=sim.device) * 0.05
            unit_v_cmd = v_cmd / v_cmd.norm()

            agent_spline = goc_mpc.last_cycle_splines[0]
            begin_time = agent_spline.begin()
            end_time = agent_spline.end()
            times = np.linspace(begin_time, end_time, 100)
            agent_xi_l, _ = agent_spline.eval_multiple(times)

            nodes_and_taus = list(zip(
                goc_mpc.timing_mpc.get_next_nodes(),
                goc_mpc.timing_mpc.get_next_taus()
            ))

            time_deltas_list = goc_mpc.timing_mpc.view_time_deltas_list()

            # rem = tuple(goc_mpc.remaining_phases)
            # if rem != prev_remaining:
            #     print("remaining_phases:", goc_mpc.remaining_phases,
            #           "completed_phases:", sorted(goc_mpc.completed_phases))
            #     prev_remaining = rem


        draw.clear_points()
        draw.draw_points(
            [(x[0], x[1], x[2]) for x in agent_xi_l[:, :3]],
            [(1, 0, 0, 1) for _ in agent_xi_l[:, :3]],
            [10 for _ in agent_xi_l[:, :3]]
        )

        draw.draw_lines([
            (x[0], x[1], x[2])
        ], [
            (x[0]+unit_v_cmd[0], x[1]+unit_v_cmd[1], x[2]+unit_v_cmd[2]),
        ], [(0, 0, 1, 1)], [10])

        # --- Convert end-effector velocity to joints velocity ----------------
        joint_vel_des = vel_controller.compute(v_cmd, jacobian)

        if nodes_and_taus:
            next_node, next_tau = nodes_and_taus[0]

            print(f"[GoC t={t:.2f}] node={next_node}, tau={next_tau:.3f}, "
                  f"remaining={goc_mpc.remaining_phases}\n"
                  f"Current pos: [{x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}]\n"
                  f"Current cmd: [{v_cmd[0]:.3f}, {v_cmd[1]:.3f}, {v_cmd[2]:.3f}]\n"
                  f"Joint vels: [{joint_vel_des}]\n"
                  f"Target waypoint 0: {goc_mpc.waypoint_mpc.view_waypoints()[0][:3]}")


        robot.set_joint_velocity_target(joint_vel_des,
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
