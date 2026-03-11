import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="b1", help="Robot to visualize")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(args).app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass

    from isaac_goc_mpc.ur5e.ur5e_robotiq_2f85 import UR5e_CFG, UR5e_ROBOTIQ_2F_85_CFG
    from isaac_goc_mpc.b1.b1 import UNITREE_B1_CFG, UNITREE_B1_WITH_Z1_CFG
    from isaac_goc_mpc.g1.g1 import UNITREE_G1_29DOF_CFG

    robot_cfgs = {
        "ur5e": UR5e_CFG,
        "ur5e_robotiq_2f_85": UR5e_ROBOTIQ_2F_85_CFG,
        "b1": UNITREE_B1_CFG,
        "b1_with_z1": UNITREE_B1_WITH_Z1_CFG,
        "g1": UNITREE_G1_29DOF_CFG,
    }

    assert args.robot in robot_cfgs, f"{args.robot} is not supported"
    robot_cfg = robot_cfgs[args.robot]

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        # ground + light must be AssetBaseCfg
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ))
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )

        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim = SimulationContext(sim_utils.SimulationCfg(
        dt=0.002,  # keep small if possible
        device=args.device
    ))
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.5])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))

    sim.reset()
    scene.reset()
    print("[OK] Spawned. If LIVESTREAM is enabled, open the WebRTC UI now.")

    sim_dt = sim.get_physics_dt()
    while app.is_running():
        sim.step()
        scene.update(sim_dt)

    app.close()

if __name__ == "__main__":
    main()

