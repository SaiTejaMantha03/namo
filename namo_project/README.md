# NAMO Deadlock Prototype

This prototype focuses on one goal only:
reproduce a narrow-corridor deadlock with two robots and one movable box.

## Environment

Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies for Apple Silicon macOS:

```bash
pip install -r requirements.txt
```

`pybullet-arm64` is used here because it is the Apple Silicon compatible drop-in package for `pybullet`.

## Verify Torch MPS

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

In this runtime, Torch reports `is_built=True` and `is_available=False`, so Metal acceleration is compiled in but not currently available to the process.

## Configurable Scenario

The scene is now driven by [scenario_config.json](</Users/saitejamantha/Documents/New project/namo_project/scenario_config.json>).

You can change:

- robot radius, height, mass, friction, and speed
- box size, mass, friction, and start cell
- corridor row and corridor wall spacing
- map size and cell size
- simulation step count and logging frequency

## Run The Deadlock Scenario

Headless:

```bash
python test_env.py --steps 900
```

GUI:

```bash
python test_env.py --gui

Use a custom config file:

```bash
python test_env.py --config scenario_config.json --gui
```
```

## Current Scenario

- 10x10 grid
- 2 robots
- 2 corridor walls
- 1 narrow passage
- 1 movable box
- goal swap across the corridor

Expected behavior:

`A ---> corridor <--- B`

The robots enter from opposite sides, contest the same passage, interact with the movable box, and get stuck.
