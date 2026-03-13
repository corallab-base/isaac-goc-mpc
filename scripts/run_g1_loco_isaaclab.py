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
    home_pose = np.array([0.00, 0.00, 0.30])
    goal1_pose = np.array([0.00, 5.00, 0.30])
    goal2_pose = np.array([5.00, 0.00, 0.30])

    A = np.eye(joint_agent_dim)  # pins all 7 dims

    graph.add_robots_linear_eq(0, A, home_pose)
    graph.add_robots_linear_eq(1, A, goal1_pose)
    graph.add_robots_linear_eq(2, A, goal2_pose)

    spline_spec = [Block.R(3)]
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
    from isaac_goc_mpc.g1.g1 import UNITREE_G1_29DOF_CFG

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
        """Configuration for the locomotion velocity-tracking environment."""

        # Scene settings
        scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5)
        # Basic settings
        observations: g1_env_cfg.ObservationsCfg = g1_env_cfg.ObservationsCfg()
        actions: g1_env_cfg.ActionsCfg = g1_env_cfg.ActionsCfg()
        commands: g1_env_cfg.CommandsCfg = g1_env_cfg.CommandsCfg()

        rewards = None
        terminations = None

        def __post_init__(self):
            """Post initialization."""
            # general settings
            self.decimation = 4
            self.episode_length_s = 20.0
            # simulation settings
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation
            # self.sim.physics_material = self.scene.terrain.physics_material
            self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

    env_cfg = G1EnvCfg()

    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    dt = env.unwrapped.step_dt

    controller = G1LocomotionController()

    obs, _ = env.reset()

    # simulate physics
    count = 0
    while app.is_running():
        with torch.inference_mode():
            action = controller.inference(obs["policy"])
            obs, rew, terminated, truncated, info = env.step(action)
            count += 1

    # close the environment
    env.close()

    # close the simulation
    app.close()

if __name__ == "__main__":
    main()
