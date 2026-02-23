import argparse
import math
import time

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    # --- Goal specification ---
    # Option 1: offset from current EE (A)
    parser.add_argument("--goal_dx", type=float, default=0.10, help="Goal offset +X (m) in WORLD.")
    parser.add_argument("--goal_dy", type=float, default=0.10, help="Goal offset +Y (m) in WORLD.")
    parser.add_argument("--goal_dz", type=float, default=0.10, help="Goal offset +Z (m) in WORLD.")

    # Option 2: explicit world position (overrides offsets if provided)
    parser.add_argument("--goal_x", type=float, default=None, help="Goal X (m) in WORLD.")
    parser.add_argument("--goal_y", type=float, default=None, help="Goal Y (m) in WORLD.")
    parser.add_argument("--goal_z", type=float, default=None, help="Goal Z (m) in WORLD.")

    # --- Timing knobs ---
    parser.add_argument("--dt", type=float, default=0.01, help="Physics dt (s).")
    parser.add_argument("--mpc_dt", type=float, default=0.20, help="How often to call GoC-MPC.step (s).")
    parser.add_argument("--lookahead", type=float, default=0.20, help="Seconds ahead to sample from GoC spline.")
    parser.add_argument("--goc_dt", type=float, default=0.10, help="GoC short_path_time_per_step (s).")

    # --- Stop condition ---
    parser.add_argument("--pos_tol", type=float, default=0.01, help="Stop when EE within this (m) of goal.")
    parser.add_argument("--max_time", type=float, default=20.0, help="Stop after this many seconds.")

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
    scene.update(sim_dt)  # populate buffers

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

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=[EE_BODY_NAME])
    robot_entity_cfg.resolve(scene)

    arm_joint_ids = robot_entity_cfg.joint_ids
    ee_body_id = robot_entity_cfg.body_ids[0]

    # Jacobian indexing detail
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    # Differential IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Define A (start) and B (goal) in WORLD
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]  # (1,7) = [x y z qw qx qy qz]
    start_pos_w = ee_pose_w[:, 0:3].clone()
    start_quat_w = ee_pose_w[:, 3:7].clone()

    if args.goal_x is not None and args.goal_y is not None and args.goal_z is not None:
        goal_pos_w = torch.tensor([[args.goal_x, args.goal_y, args.goal_z]], device=sim.device, dtype=torch.float32)
    else:
        goal_pos_w = start_pos_w + torch.tensor([[args.goal_dx, args.goal_dy, args.goal_dz]], device=sim.device)

    # keep orientation fixed for this first A->B test
    goal_quat_w = start_quat_w.clone()

    print("[OK] Spawned robot.")
    print("EE body:", EE_BODY_NAME, " (body_id:", int(ee_body_id), " jacobi_idx:", int(ee_jacobi_idx), ")")
    print("Arm joints:", ARM_JOINT_NAMES)
    print("A start pos (w):", start_pos_w[0].tolist())
    print("B goal  pos (w):", goal_pos_w[0].tolist())

    # ----------------------------
    # Build minimal GoC graph (A -> B) for a single free-body agent
    # ----------------------------
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser_drake = Parser(plant)

    free_body_urdf = files(goc_pkg) / "descriptions" / "free_body_6dof.urdf"
    model_instances = parser_drake.AddModels(str(free_body_urdf))
    plant.RenameModelInstance(model_instances[0], "free_body_0")
    plant.Finalize()

    symbolic_plant = plant.ToSymbolic()

    state_lower_bound = -10.0
    state_upper_bound = +10.0

    graph = GraphOfConstraints(symbolic_plant, ["free_body_0"], [], state_lower_bound, state_upper_bound)

    # canonical minimal structure: 2 nodes + 1 edge
    graph.structure.add_nodes(2)
    graph.structure.add_edge(0, 1, True)

    joint_agent_dim = graph.num_agents * graph.dim  # should be 1*7 = 7

    A = torch.cat([ee_pose_w[0], ee_quat0_w[0]], dim=0).detach().cpu().numpy().astype(float)  # (7,)
    B = torch.cat([goal_pose_w[0], goal_quat_w[0]], dim=0).detach().cpu().numpy().astype(float)  # (7,)

    graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), A)
    graph.add_robots_linear_eq(1, np.eye(joint_agent_dim), B)

    spline_spec = [Block.R(3), Block.SO3()]

    # This dt is the "discretization" GoC uses for xi_h (matches how the example steps)
    GOC_DT = 0.10  # feel free to tune

    goc = GraphOfConstraintsMPC(
        graph,
        spline_spec,
        short_path_time_per_step=GOC_DT,
        solve_for_waypoints_once=False,
        time_delta_cutoff=0.0,
        phi_tolerance=0.02,
    )

    goc.reset()
 
    # ----------------------------
    # Main loop
    # ----------------------------
    sim_time = 0.0
    step_count = 0

    # how many physics steps per GoC update
    goc_stride = max(1, int(round(GOC_DT / sim_dt)))
    last_xi_h = None

    # how many GoC discrete steps ahead to command (lookahead seconds)
    k_ahead = max(1, int(round(args.lookahead / GOC_DT)))

    while app.is_running():
        # Current EE pose in world
        ee_pose_w_now = robot.data.body_pose_w[:, ee_body_id]
        x = ee_pose_w_now[0].detach().cpu().numpy().astype(float)      # (7,)
        x_dot = np.zeros(6, dtype=float)                               # (6,)

        # Update GoC every goc_stride frames
        if step_count % goc_stride == 0 or last_xi_h is None:
            try:
                xi_h, _, _ = goc.step(sim_time, x, x_dot, teleport=False)
                last_xi_h = xi_h
            except RuntimeError as e:
                # fall back to last cycle if GoC throws
                try:
                    xi_h, _, _ = goc.last_cycle_short_path
                    last_xi_h = xi_h
                    print("[WARN] GoC step failed, using last_cycle_short_path:", repr(e))
                except Exception:
                    print("[WARN] GoC step failed and no fallback available:", repr(e))
                    last_xi_h = None

        if last_xi_h is None or last_xi_h.shape[0] == 0:
            # nothing from GoC yet — hold current
            des_pose_w = ee_pose_w_now.clone()
        else:
            idx = min(k_ahead, last_xi_h.shape[0] - 1)
            q_des = np.array(last_xi_h[idx], dtype=float).copy()  # (7,) = [x y z qw qx qy qz] in this codepath
    
            # normalize quaternion (good hygiene)
            nq = np.linalg.norm(q_des[3:7]) + 1e-9
            q_des[3:7] /= nq

            des_pose_w = torch.tensor(q_des, device=sim.device, dtype=torch.float32).unsqueeze(0)

        # Convert desired pose (world) -> desired pose in robot root frame for DiffIK
        root_pose_w = robot.data.root_pose_w
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            des_pose_w[:, 0:3], des_pose_w[:, 3:7],
        )
        ik_cmd = torch.cat([goal_pos_b, goal_quat_b], dim=-1)
        diff_ik.set_command(ik_cmd)

        # Jacobian + current EE pose in base frame
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w_now[:, 0:3], ee_pose_w_now[:, 3:7],
        )
        joint_pos = robot.data.joint_pos[:, arm_joint_ids]
        joint_pos_des = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)

        # 1 Hz debug: are we commanding something different?
        if step_count % max(1, int(round(1.0 / sim_dt))) == 0:
            pos_err = torch.norm(des_pose_w[:, 0:3] - ee_pose_w_now[:, 0:3]).item()
            print(f"t={sim_time:6.2f}  pos_err={pos_err:.3f}  des={des_pose_w[0,0:3].tolist()}  cur={ee_pose_w_now[0,0:3].tolist()}")

        # Step sim
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        sim_time += sim_dt
        step_count += 1

    app.close()



if __name__ == "__main__":
    main()
