# Isaac GoC MPC

## Running scripts (IsaacLab)

This repo assumes IsaacLab is installed separately and you run scripts through the provided launch wrappers. 

### With WebRTC streaming
```bash
isaaclab -p scripts/view_robot.py
```

### Headless / no streaming
```bash
isaaclab_nostream -p scripts/print_robot_names.py --headless
```

## Environment 
This repo uses .envrc to set:
- ```ISAACLAB_DIR``` (path to a pre-installed IsaacLab)
- Drake / GoC-MPC env vars
- optional livestream defaults (WebRTC)

If you don't use direnv, manually source your IsaacLab venv and export the same vars. 


