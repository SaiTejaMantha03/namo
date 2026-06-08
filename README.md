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

The scene is now driven by [scenario_config.json](./simulation/scenario_config.json).

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

## Results Section

This section demonstrates the capabilities of our NAMO (Navigation Among Movable Obstacles) evaluation pipeline.

### 1. Heatmap Figures
These figures highlight the cost distribution across different environments:
*   ![Warehouse Heatmap](file:///Users/saitejamantha/Documents/New%20project/results/heatmaps/warehouse_heatmap.png)

### 2. Decision Outputs
The trained UNet predictor directly outputs decision heuristics (Bypass vs Removal) without the need to plan full paths to compare costs:
*   ![3x3 Bypass Decision](file:///Users/saitejamantha/Documents/New%20project/results/visualizations/namo_3x3_bypass.png)
*   ![3x3 Removal Decision](file:///Users/saitejamantha/Documents/New%20project/results/visualizations/namo_3x3_removal.png)
*   ![5x5 Bypass Decision](file:///Users/saitejamantha/Documents/New%20project/results/visualizations/namo_5x5_bypass.png)
*   ![5x5 Removal Decision](file:///Users/saitejamantha/Documents/New%20project/results/visualizations/namo_5x5_removal.png)

### 3. Deadlock Examples
When robots follow selfish paths in narrow corridors, they inevitably deadlock. We identify deadlock situations by predicting collisions and excessive push costs in constrained geometries, enabling proactive yielding or coordinated movable obstacle displacement.
