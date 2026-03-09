import torch
import gymnasium as gym

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_obs_groups

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

# Workaround for module starting with number (stupid!)
from importlib import import_module
g1_env_cfg = import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")



POLICY_PATH = "/home/azucek/azucek/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-03-04_23-45-35/model_13600.pt"


class G1LocomotionController:
    def __init__(self, device="cuda"):
        self.device = device

        agent_cfg = BasePPORunnerCfg()

        # annoyingly, constructing the env is the simplest way to configure the
        # PPO actor.
        env_cfg = g1_env_cfg.RobotPlayEnvCfg()
        env = gym.make("Unitree-G1-29dof-Velocity", cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        obs = env.get_observations()
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        actor_critic = ActorCritic(
            obs,
            obs_groups,
            env.num_actions,
            **agent_cfg.policy.to_dict()
        )

        loaded_dict = torch.load(POLICY_PATH)
        ac_params = loaded_dict["model_state_dict"]
        actor_critic.load_state_dict(ac_params)

        self.actor_critic = actor_critic.to(self.device)
        self.actor_critic.eval()

    def compute_obs(self, robot, commands):
        raise NotImplementedError()

    def get_action(self, obs):
        raise NotImplementedError()
