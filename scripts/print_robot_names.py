import argparse
import os

from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # MUST create the SimulationApp BEFORE importing most Isaac/Omniverse modules
    app = AppLauncher(args).app

    # imports AFTER app is created
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from isaac_goc_mpc.ur5e.ur5e_robotiq_2f85 import UR5e_ROBOTIQ_2F_85_CFG as ROBOT_CFG

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))

    sim.reset()
    scene.reset()

    robot = scene["robot"]
    print("USD:", os.environ.get("UR5E_GRIPPER_USD"))
    print("JOINT NAMES:", robot.data.joint_names)
    print("BODY NAMES:", robot.data.body_names)

    app.close()


if __name__ == "__main__":
    main()

