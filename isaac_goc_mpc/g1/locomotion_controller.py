import torch
import gymnasium as gym
from tensordict import TensorDict
from collections import deque

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_obs_groups

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


POLICY_PATH = "/home/azucek/azucek/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-03-04_23-45-35/model_13600.pt"


class G1LocomotionController:
    def __init__(self, device="cuda"):
        self.device = device

        agent_cfg = BasePPORunnerCfg()

        obs = TensorDict({
            "policy": torch.zeros((1, 480,)),
            "critic": torch.zeros((1, 495,))
        }, batch_size=[1])
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        action_dim = 29

        actor_critic = ActorCritic(
            obs,
            obs_groups,
            action_dim,
            **agent_cfg.policy.to_dict()
        )

        loaded_dict = torch.load(POLICY_PATH)
        ac_params = loaded_dict["model_state_dict"]
        actor_critic.load_state_dict(ac_params)

        self.actor_critic = actor_critic.to(self.device)
        self.actor_critic.eval()

        # TODO: get from config
        self.history_length = 5
        self.last_action = torch.zeros((action_dim,), device=self.device)
        self.obs_history = deque([], self.history_length)
        self.clip_actions = torch.ones((action_dim,), device=self.device) * 10.0

    def get_single_step_obs(self, robot, command):
        """Extracts and scales a single time-step observation."""

        # base_ang_vel (b, 3,)
        base_ang_vel = robot.data.root_ang_vel_b
        b = base_ang_vel.shape[0]

        # projected_gravity (b, 3,)
        proj_gravity = robot.data.projected_gravity_b

        # velocity_command (b, 3,) - [x_vel, y_vel, yaw_vel]
        vel_cmd = torch.tensor(command, device=self.device).tile((b, 1))

        # joint_pos_rel (b, 29,)
        joint_pos_rel = robot.data.joint_pos[:, :] - \
            robot.data.default_joint_pos[:, :]

        # joint_vel_rel (b, 29,)
        joint_vel_rel = robot.data.joint_vel[:, :] - \
            robot.data.default_joint_vel[:, :]

        # Concatenate in the EXACT order of your PolicyCfg
        current_obs = torch.cat([
            base_ang_vel,   # 3
            proj_gravity,   # 3
            vel_cmd,        # 3
            joint_pos_rel,  # 29
            joint_vel_rel,  # 29
            self.last_action.tile((b, 1)) # 29
        ], dim=-1)

        return current_obs

    def advance(self, robot, command):
        # 1. Get latest obs
        current_obs = self.get_single_step_obs(robot, command)

        # 2. Update history
        if len(self.obs_history) == 0:
            for _ in range(self.history_length):
                self.obs_history.append(current_obs)
        else:
            self.obs_history.append(current_obs)

        # 3. Flatten history for policy: [Step T-4, T-3, T-2, T-1, T]
        policy_input = torch.cat(list(self.obs_history), dim=-1)

        # 4. Inference
        with torch.inference_mode():
            policy_input = self.actor_critic.actor_obs_normalizer(policy_input)
            if self.actor_critic.state_dependent_std:
                actions_raw = self.actor_critic.actor(policy_input)[..., 0, :]
            else:
                actions_raw = self.actor_critic.actor(policy_input)

        scaled_actions = actions_raw * 0.25
        target_joint_pos = robot.data.default_joint_pos + scaled_actions

        target_joint_pos = torch.clamp(target_joint_pos, -100.0, 100.0)

        self.last_action = target_joint_pos
        return target_joint_pos

    def inference(self, obs):
        policy_input = self.actor_critic.actor_obs_normalizer(obs)
        if self.actor_critic.state_dependent_std:
            actions = self.actor_critic.actor(policy_input)[..., 0, :]
        else:
            actions = self.actor_critic.actor(policy_input)
        return actions
