import argparse
import math
import time
import numpy as np
import torch

from isaaclab.app import AppLauncher

def setup_goc_mpc():
    import numpy as np

    from goc_mpc.goc_mpc import GraphOfConstraintsMPC
    from goc_mpc.splines import Block
    from goc_mpc._ext.goc_mpc import GraphOfConstraints
    from goc_mpc.simple_drake_env import SimpleDrakeGym

    # switch to using a point mass robot, ignore orientation
    env = SimpleDrakeGym(["point_mass_0"], [], meshcat=None)

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    # the api changed here, it no longer takes a plant.
    graph = GraphOfConstraints(["point_mass"], [],
                               state_lower_bound, state_upper_bound)

    agent_dim = graph.dim;
    joint_agent_dim = graph.num_agents * graph.dim

    # Construct graph
    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    # Identity quaternion for stability (wxyz), changed to match .8
    home_pose = np.array([0.00, 0.00, 0.80])
    goal1_pose = np.array([0.00, 5.00, 0.80])
    goal2_pose = np.array([5.00, 0.00, 0.80])

    A = np.eye(joint_agent_dim)  # pins all 3 dims

    #prints for graph
    print("graph.dim:", graph.dim)
    print("graph.num_agents:", graph.num_agents)
    print("joint_agent_dim:", joint_agent_dim)
    print("A.shape:", A.shape)
    print("home_pose.shape:", home_pose.shape)
    print("goal1_pose.shape:", goal1_pose.shape)
    print("goal2_pose.shape:", goal2_pose.shape)


    placeholder_id = 0
    #10 just placeholder for robot id i guess
    # changed from add_robot_ bc I need real distance checks, not just returning 0 for obs check
    graph.add_robot_linear_eq_dist(0, placeholder_id, A, home_pose)
    graph.add_robot_linear_eq_dist(1, placeholder_id, A, goal1_pose)
    graph.add_robot_linear_eq_dist(2, placeholder_id, A, goal2_pose)

    spline_spec = [Block.R(3)]
    goc_mpc = GraphOfConstraintsMPC(
        graph, spline_spec,
        short_path_time_per_step=0.1,
        solve_for_waypoints_once=False,
        time_delta_cutoff=0.0,
        phi_tolerance=0.03,
        max_acc=15.0,
    )
    return env, graph, goc_mpc

def main():


    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    # --- TODO: Add optional args here ---

    # ---
    args = parser.parse_args()
    app = AppLauncher(args).app


    # Isaac imports AFTER app is created
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.utils.math import subtract_frame_transforms

    from isaac_goc_mpc.g1.locomotion_controller import G1LocomotionController

    # Robot cfg
    from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
    #from isaac_goc_mpc.g1.g1 import UNITREE_G1_29DOF_CFG

    # GoC-MPC + Drake imports
    from goc_mpc.goc_mpc import GraphOfConstraintsMPC
    from goc_mpc.splines import Block
    from goc_mpc._ext.goc_mpc import GraphOfConstraints

    from pydrake.systems.framework import DiagramBuilder
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.multibody.parsing import Parser
    from importlib.resources import files
    import goc_mpc as goc_pkg

    # Workaround for module starting with number (stupid!)
    from importlib import import_module
    g1_env_cfg = import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")
    
    # GoC stuff...
    _, _, goc_mpc = setup_goc_mpc()

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
        robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



    @configclass
    class G1EnvCfg(ManagerBasedRLEnvCfg):
        scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5)

        observations: g1_env_cfg.ObservationsCfg = g1_env_cfg.ObservationsCfg()
        actions: g1_env_cfg.ActionsCfg = g1_env_cfg.ActionsCfg()
        commands: g1_env_cfg.CommandsCfg = g1_env_cfg.CommandsCfg()
        events: g1_env_cfg.EventCfg = g1_env_cfg.EventCfg()

        rewards = None
        terminations = None

        def __post_init__(self):
            self.decimation = 4
            self.episode_length_s = 10000.0
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation
            self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

            self.events.push_robot = None
            self.events.base_external_force_torque = None

    """
    @configclass
    class G1EnvCfg(ManagerBasedRLEnvCfg):
        #Configuration for the locomotion velocity-tracking environment.

        # Scene settings
        scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5)
        # Basic settings
        observations: g1_env_cfg.ObservationsCfg = g1_env_cfg.ObservationsCfg()
        actions: g1_env_cfg.ActionsCfg = g1_env_cfg.ActionsCfg()
        commands: g1_env_cfg.CommandsCfg = g1_env_cfg.CommandsCfg()

        rewards = None
        terminations = None

        def __post_init__(self):
            #Post initialization.
            # general settings
            self.decimation = 4
            self.episode_length_s = 20.0
            # simulation settings
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation
            # self.sim.physics_material = self.scene.terrain.physics_material
            self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
    """
    env_cfg = G1EnvCfg()

    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    dt = env.unwrapped.step_dt

    controller = G1LocomotionController()

    obs, _ = env.reset()

    # simulate physics
    count = 0


    print(env.command_manager)

    for name, term in env.command_manager._terms.items():
        print("TERM:", name, type(term))
        if hasattr(term, "command"):
            print("  command shape:", term.command.shape)

    while app.is_running():
        with torch.inference_mode():

            robot = env.scene["robot"]
            t = count * dt
            x = robot.data.root_pos_w[0].cpu().numpy()
            x_dot = robot.data.root_lin_vel_w[0].cpu().numpy()


            # print("t:", t, type(t))
            # print("x:", x, x.shape, x.dtype)
            # print("x_dot:", x_dot, x_dot.shape, x_dot.dtype)


            # print("CALLING GOC_MPC STEP FOR THE: ", count, " time!\n\n")

            # positions, velocities, times = goc_mpc.step(t, x, x_dot)

            # print("\n#####GoC Inputs#####")
            # print("t: ", t, " x: ", x, "x_dot: ", x_dot)


            # print("\n#####GoC Outputs#####")
            # print("positions: ", positions)
            # print("velocities: ", velocities)
            # print("times: ", times)

            # for i in range(5):

            #     x_vel = 0
            #     y_vel = -.1
            #     yaw_vel = 0

            #     #for proof of concept, take way to big goc vels, and clamp/normalize to reasonable vals
            #     #x_vel = float(np.clip(velocities[1][0], -.3, .3))
            #     #y_vel = float(np.clip(velocities[1][1], -.3, .3))


            #     #use raw values:
            #     #x_vel = velocities[1][0]
            #     #y_vel = velocities[1][1]

            #     if i == 9:
            #         #yaw_vel = 0
            #         base_idx = i*96
            #         obs["policy"][:, base_idx+6] = x_vel
            #         obs["policy"][:, base_idx+7] = y_vel
            #         obs["policy"][:, base_idx+8] = yaw_vel

            cmd_term = env.command_manager._terms["base_velocity"]
            cmd_term.command[:] = torch.tensor([[.1, 0, 0]], device=cmd_term.command.device)

            action = controller.inference(obs["policy"])
            
            cmd_term = env.command_manager._terms["base_velocity"]

            # print("before set:", cmd_term.command[0].cpu().numpy())
            cmd_term.command[:] = torch.tensor([[0.3, 0, 0.0]], device=cmd_term.command.device)
            # print("after set:", cmd_term.command[0].cpu().numpy())

            action = controller.inference(obs["policy"])
            obs, rew, terminated, truncated, info = env.step(action)

            # print("after step cmd:", cmd_term.command[0].cpu().numpy())
            # print("terminated:", terminated, "truncated:", truncated)

            #command = [x_vel, y_vel, yaw_vel]
            #action = controller.advance(robot, command)
            obs, rew, terminated, truncated, info = env.step(action)
            count += 1

    # close the environment
    env.close()

    # close the simulation
    app.close()

if __name__ == "__main__":
    main()
