import argparse
import math

from isaaclab.app import AppLauncher


def build_minimal_pose_goal_goc(goal_pose_w: "np.ndarray", short_path_dt: float):
    """
    Build the *smallest* GoC graph that produces a pose spline for a single 6DOF free body:
      node 0: trivial constraint (so the node isn't empty)
      node 1: pose == goal_pose_w
      edge 0 -> 1

    goal_pose_w is (7,) in WORLD frame: [x, y, z, qw, qx, qy, qz]
    """
    import numpy as np
    from goc_mpc.splines import Block
    from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
    from goc_mpc.simple_drake_env import SimpleDrakeGym

    agents = ["free_body_0"]
    objects = []

    # Drake plant (used only to back the GoC graph)
    env = SimpleDrakeGym(agents, objects)
    symbolic_plant = env.plant.ToSymbolic()

    # IMPORTANT: GraphOfConstraints wants *scalar* bounds, not per-dimension arrays
    state_lower_bound = -10.0
    state_upper_bound = +10.0

    graph = GraphOfConstraints(symbolic_plant, agents, objects, state_lower_bound, state_upper_bound)

    # Two nodes, one directed edge
    graph.structure.add_nodes(2)
    graph.structure.add_edge(0, 1, True)

    joint_agent_dim = graph.num_agents * graph.dim  # for a single free body: 1 * 7 = 7

    # Node 0: trivial constraint (does nothing, but avoids "empty node" weirdness)
    graph.add_robots_linear_eq(0, np.zeros((1, joint_agent_dim)), np.zeros((1,)))

    # Node 1: hard pose goal
    graph.add_robots_linear_eq(1, np.eye(joint_agent_dim), goal_pose_w)

    spline_spec = [Block.R(3), Block.SO3()]

    goc_mpc = GraphOfConstraintsMPC(
        graph,
        spline_spec,
        short_path_time_per_step=short_path_dt,
        # These knobs are optional; keep defaults unless you need them.
        solve_for_waypoints_once=False,
        time_delta_cutoff=0.0,
        phi_tolerance=0.03,
    )

    return goc_mpc


def _spline_eval_pose(spline, t_query: float):
    """
    Returns ambient pose q(t) = [pos(3), quat(4)] for a GoC spline.
    Works with either:
      - spline.eval(t) -> (q, qdot, qddot)
      - spline.eval(t, deriv) style APIs (if present)
    """
    out = spline.eval(t_query)
    # Most bindings return (q, qdot, qddot)
    if isinstance(out, tuple) or isinstance(out, list):
        return out[0]
    return out


def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    # --- Goal offset in WORLD frame ---
    parser.add_argument("--goal_dx", type=float, default=0.10)
    parser.add_argument("--goal_dy", type=float, default=0.00)
    parser.add_argument("--goal_dz", type=float, default=0.00)

    # --- Timing knobs ---
    parser.add_argument("--lookahead", type=float, default=0.20, help="Seconds ahead when sampling the GoC spline.")
    parser.add_argument("--mpc_dt", type=float, default=0.10, help="How often to call goc_mpc.step (seconds).")
    parser.add_argument("--dt", type=float, default=0.01, help="Physics dt (seconds).")

    args = parser.parse_args()
    app = AppLauncher(args).app

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

    from isaac_goc_mpc.ur5e.ur5e_robotiq_2f85 import UR5e_ROBOTIQ_2F_85_CFG 

    # ----------------------------
    # Scene configuration
    # ----------------------------
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        robot: ArticulationCfg = UR5e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.5])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))

    sim.reset()
    scene.reset()

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # Populate buffers
    scene.update(sim_dt)

    # ----------------------------
    # Hard-coded EE + arm joints (as discussed)
    # ----------------------------
    EE_BODY_NAME = "wrist_3_link"
    ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=[EE_BODY_NAME])

    try:
        robot_entity_cfg.resolve(scene)
    except Exception:
        print(f"\n[ERROR] Could not resolve EE body name: {EE_BODY_NAME}")
        print("Available body names:")
        for n in list(robot.data.body_names):
            print("  -", n)
        raise

    arm_joint_ids = robot_entity_cfg.joint_ids
    ee_body_id = robot_entity_cfg.body_ids[0]

    # Jacobian indexing detail (IsaacLab convention)
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    # ----------------------------
    # Differential IK controller
    # ----------------------------
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Current EE pose in WORLD
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]  # (N, 7) = [x y z qw qx qy qz]
    ee_pos0_w = ee_pose_w[:, 0:3].clone()
    ee_quat0_w = ee_pose_w[:, 3:7].clone()

    goal_pos_w = ee_pos0_w + torch.tensor([[args.goal_dx, args.goal_dy, args.goal_dz]], device=sim.device)
    goal_quat_w = ee_quat0_w.clone()  # keep orientation fixed for this first test

    print("[OK] Spawned robot.")
    print("EE body:", EE_BODY_NAME, " (body_id:", int(ee_body_id), " jacobi_idx:", int(ee_jacobi_idx), ")")
    print("Arm joints:", ARM_JOINT_NAMES)
    print("Initial EE pos (w):", ee_pos0_w[0].tolist())
    print("Goal EE pos    (w):", goal_pos_w[0].tolist())

    # ----------------------------
    # Build minimal GoC-MPC: 1 free-body, 2-node graph, pose goal at node 1
    # ----------------------------
    goal_pose_w_np = np.concatenate(
        [goal_pos_w[0].detach().cpu().numpy(), goal_quat_w[0].detach().cpu().numpy()],
        axis=0,
    ).astype(float)

    goc_mpc = build_minimal_pose_goal_goc(goal_pose_w_np, short_path_dt=args.mpc_dt)
    goc_mpc.reset()

    # ----------------------------
    # Main loop
    # ----------------------------
    sim_time = 0.0
    last_mpc_solve_t = -1e9

    # Cached desired pose (WORLD)
    des_pose_w = ee_pose_w.clone()  # (1,7)

    while app.is_running():
        # Current EE pose in world
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        x = ee_pose_w[0].detach().cpu().numpy().astype(float)  # (7,)

        # We don't have a measured twist yet; zeros is fine for this first integration
        x_dot = np.zeros(6, dtype=float)

        # Re-solve GoC-MPC occasionally
        if sim_time - last_mpc_solve_t >= args.mpc_dt:
            try:
                goc_mpc.step(sim_time, x, x_dot, teleport=False)
                last_mpc_solve_t = sim_time
            except Exception as e:
                print("[WARN] goc_mpc.step failed:", repr(e))

        # Sample the latest pose spline
        try:
            spline = goc_mpc.last_cycle_splines[0]
            t_begin = float(spline.begin())
            t_end = float(spline.end())
            t_query = sim_time + args.lookahead
            t_query = max(t_begin, min(t_end, t_query))

            q_ambient = np.asarray(_spline_eval_pose(spline, t_query), dtype=float)  # (7,)
            des_pose_w = torch.tensor(q_ambient, device=sim.device, dtype=torch.float32).unsqueeze(0)
        except Exception:
            # If spline isn't ready yet, just hold current
            des_pose_w = ee_pose_w.clone()

        # Desired pose for DiffIK must be in the robot root frame
        root_pose_w = robot.data.root_pose_w  # (N,7)
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], des_pose_w[:, 0:3], des_pose_w[:, 3:7]
        )
        ik_cmd = torch.cat([goal_pos_b, goal_quat_b], dim=-1)
        diff_ik.set_command(ik_cmd)

        # Jacobian + current EE pose in base frame
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]  # (N,6,6)
        ee_pose_w_now = robot.data.body_pose_w[:, ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w_now[:, 0:3], ee_pose_w_now[:, 3:7]
        )
        joint_pos = robot.data.joint_pos[:, arm_joint_ids]

        joint_pos_des = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # Apply joint targets
        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)

        # Step sim
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt

    app.close()


if __name__ == "__main__":
    main()
