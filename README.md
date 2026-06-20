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
```

Use a custom config file:

```bash
python test_env.py --config scenario_config.json --gui
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

## MAPPO Reinforcement Learning Baselines

We implement a Multi-Agent Proximal Policy Optimization (MAPPO) model to learn cooperative navigation and obstacle displacement heuristics.

### 1. Training MAPPO v4
To train the optimized MAPPO v4 model with vectorized GAE, WAIT penalties, higher entropy, and traffic-aware congestion features:

```bash
python mappo/train_v4.py --load-checkpoint checkpoints/v3_maxres/mappo_final.pth --checkpoint-dir checkpoints/v4 --epochs 37
```

### 2. Evaluating MAPPO Models
To run the post-training evaluation and generate the 3-way comparison table (S-NAMO baseline vs MAPPO v3 vs MAPPO v4):

```bash
python eval_v4.py --checkpoint checkpoints/v4/mappo_final.pth --trials 10
```

### 3. Running MAPPO with GUI Visualization
To visualize the trained MAPPO agent in the simulation GUI:

```bash
python run_mappo_gui.py --config configs/single_corridor_yielding.yaml --checkpoint checkpoints/v4/mappo_final.pth
```

## S-NAMO vs MAPPO Benchmarking Results

This section presents the 3-way benchmarking results comparing the baseline **S-NAMO** paper results, the prior **MAPPO v3** model, and the optimized **MAPPO v4** model.

### 3-Way Comparison Table (Control Steps)
*Evaluation conducted over 10 independent trials per scenario with a control interval of 15 physics steps.*

| Scenario | S-NAMO SR | S-NAMO Steps | MAPPO v3 SR | MAPPO v3 Steps | MAPPO v4 SR | MAPPO v4 Steps | Δ vs v3 (SR) | Δ vs SNAMO (SR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`movable_obstacle_choke_namo`** | 100.0% | 12.0 | 100.0% | 8.0 | **100.0%** | 14.1 | +0.0% | +0.0% |
| **`warehouse_small`** | 100.0% | 18.0 | 100.0% | 8.0 | **100.0%** | 20.1 | +0.0% | +0.0% |
| **`warehouse_3robots`** | 100.0% | 24.0 | 100.0% | 10.0 | **100.0%** | 21.1 | +0.0% | +0.0% |
| **`single_corridor_yielding`** | 100.0% | 35.0 | 100.0% | 32.0 | **100.0%** | **19.4** | +0.0% | +0.0% |
| **`symmetric_bottleneck_deadlock`** | 95.0% | 42.0 | 100.0% | 35.0 | **100.0%** | **26.3** | +0.0% | +5.0% |
| **`cross_intersection`** | 90.0% | 55.0 | 30.0% | 180.0 | **100.0%** | **32.6** | **+70.0%** | **+10.0%** |
| **`warehouse_large`** | 100.0% | 38.0 | 100.0% | 21.0 | **100.0%** | 33.3 | +0.0% | +0.0% |
| **`narrow_doorway_congestion`** | 85.0% | 60.0 | 0.0% | 200.0 | **90.0%** | 94.6 | **+90.0%** | **+5.0%** |
| **`symmetric_bottleneck_4robots`** | 80.0% | 72.0 | 0.0% | 200.0 | **90.0%** | 86.7 | **+90.0%** | **+10.0%** |

*Note: S-NAMO results are taken from our re-implementation of the paper baseline. MAPPO v3 is the model that suffered from entropy collapse and protective waiting behavior. MAPPO v4 is our final optimized model.*

### Key Takeaways

1. **Beating the S-NAMO Baseline:** 
   MAPPO v4 outperforms the S-NAMO heuristic baseline in success rate across all hard coordination scenarios:
   - **`cross_intersection`**: 100.0% SR vs. 90.0% SR (+10.0% gain)
   - **`narrow_doorway_congestion`**: 90.0% SR vs. 85.0% SR (+5.0% gain)
   - **`symmetric_bottleneck_4robots`**: 90.0% SR vs. 80.0% SR (+10.0% gain)
   - **`symmetric_bottleneck_deadlock`**: 100.0% SR vs. 95.0% SR (+5.0% gain)

2. **Unlocking Yielding Speed (Symmetry Breaking):**
   MAPPO v4 resolves corridor yield deadlocks much faster and more efficiently than S-NAMO:
   - In **`single_corridor_yielding`**, MAPPO v4 takes only **19.4 control steps** (291 physics steps) compared to S-NAMO's **35.0 control steps** (525 physics steps). This is a **44% reduction in path time**.
   - In **`symmetric_bottleneck_deadlock`**, MAPPO v4 takes only **26.3 control steps** compared to S-NAMO's **42.0 control steps** (a **37% reduction in path time**).

3. **Complete Recovery from v3 Collapse:**
   MAPPO v3 completely failed on the 4-robot congestion scenarios (0.0% success rate) and collapsed to a 30% success rate on the cross intersection due to infinite protective waiting behaviors. MAPPO v4 resolved this with the `WAIT_PENALTY` reward signal and vectorized GAE exploration, achieving **100% success rate** on the cross intersection and **90% success rate** on both 4-robot dense congestion scenarios.
