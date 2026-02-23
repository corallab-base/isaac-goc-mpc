import argparse
import math

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    # optional tuning knobs
    parser.add_argument("--joint", type=str, default="shoulder_pan_joint")
    parser.add_argument("--amp", type=float, default=0.5, help="Amplitude (rad) for sinusoid")
    parser.add_argument("--freq", type=float, default=0.2, help="Frequency (Hz) for sinusoid")
    args = parser.parse_args()

    app = AppLauncher(args).app

    # Isaac imports AFTER app is created
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass

    from isaac_goc_mpc.ur5e.ur5e_robotiq_2f85 import UR5e_ROBOTIQ_2F_85_CFG 

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        robot: ArticulationCfg = UR5e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sim + scene
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.5])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))

    sim.reset()
    scene.reset()

    robot = scene["robot"]

    # resolve joint id
    joint_names = list(robot.data.joint_names)
    if args.joint not in joint_names:
        raise RuntimeError(
            f"Joint '{args.joint}' not found.\nAvailable joints:\n  " + "\n  ".join(joint_names)
        )
    j = joint_names.index(args.joint)
    base_val = float(robot.data.default_joint_pos[0, j].item())

    print("[OK] Spawned robot.")
    print("Commanding joint:", args.joint, " (index:", j, ")")
    print("Base position (rad):", base_val)
    print("Amp (rad):", args.amp, " Freq (Hz):", args.freq)

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0

    # main loop
    while app.is_running():
        # build joint targets: hold everything at default, wiggle one joint
        targets = robot.data.default_joint_pos.clone()
        targets[:, j] = base_val + args.amp * math.sin(2.0 * math.pi * args.freq * sim_time)

        robot.set_joint_position_target(targets)

        # write commands -> step -> update buffers
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        sim_time += sim_dt
        step_count += 1

        # lightweight logging
        if step_count % 120 == 0:
            q_now = float(robot.data.joint_pos[0, j].item())
            print(f"t={sim_time:6.2f}s  {args.joint}: q={q_now:+.3f} rad")

    app.close()


if __name__ == "__main__":
    main()
