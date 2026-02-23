import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(args).app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass

    from isaac_goc_mpc.ur5e.ur5e_robotiq_2f85 import UR5e_ROBOTIQ_2F_85_CFG 


    @configclass
    class SceneCfg(InteractiveSceneCfg):
        # ground + light must be AssetBaseCfg
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )

        robot: ArticulationCfg = UR5e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
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

