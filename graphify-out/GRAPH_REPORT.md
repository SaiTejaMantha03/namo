# Graph Report - New project  (2026-06-09)

## Corpus Check
- 29 files · ~25,803 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 211 nodes · 364 edges · 23 communities (21 shown, 2 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e960b5f0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]

## God Nodes (most connected - your core abstractions)
1. `UNetDecisionPipeline` - 22 edges
2. `NAMOmappoEnv` - 17 edges
3. `NAMOEnvironment` - 17 edges
4. `simulate_env()` - 17 edges
5. `WarehouseEnvironment` - 16 edges
6. `Path` - 14 edges
7. `robot` - 10 edges
8. `run_simulation()` - 9 edges
9. `run_simulation_baseline()` - 9 edges
10. `AttentionUNet` - 9 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `Path`  [INFERRED]
  decision/namo_visualizer.py → simulation/test_env.py
- `UNetDecisionPipeline` --uses--> `IntersectionEnvironment`  [INFERRED]
  decision/unet_decision_pipeline.py → maps/namo_environments.py
- `UNetDecisionPipeline` --uses--> `WarehouseEnvironment`  [INFERRED]
  decision/unet_decision_pipeline.py → maps/namo_environments.py
- `UNetDecisionPipeline` --uses--> `AttentionUNet`  [INFERRED]
  decision/unet_decision_pipeline.py → unet/unet.py
- `NAMOmappoEnv` --uses--> `UNetDecisionPipeline`  [INFERRED]
  mappo/mappo_env.py → decision/unet_decision_pipeline.py

## Import Cycles
- None detected.

## Communities (23 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (34): box, friction, half_extent, height, mass, start_cell, corridor, left_opening_x (+26 more)

### Community 1 - "Community 1"
Cohesion: 0.14
Nodes (10): ActorNetwork, CriticNetwork, MAPPOAgent, run_test(), a_star_internal(), find_clearing_direction_multi(), NAMOmappoEnv, Generates 20x20 risk heatmap from current layout for a specific agent. (+2 more)

### Community 2 - "Community 2"
Cohesion: 0.21
Nodes (17): Generic decision pipeline supporting any size grid:         - Automatically pads, run_all_evaluations(), UNetDecisionPipeline, Path, a_star_internal(), cell_center(), create_box(), create_robot() (+9 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (9): Dataset, NAMODataset, train(), AttentionGate, AttentionUNet, DoubleConv, Attention Gate to filter skip connection features.     g: gating signal (coarser, Attention U-Net architecture tailored for NAMO occupancy grids.     Input: (batc (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.19
Nodes (7): generate_demos(), IntersectionEnvironment, NAMOEnvironment, Dynamically adds a robot with a specified start and goal position., Generates the 2D numpy occupancy grid., Generates a beautiful Matplotlib visualization showing layout, robots, goals, an, WarehouseEnvironment

### Community 5 - "Community 5"
Cohesion: 0.32
Nodes (11): main(), cell_center(), create_box(), create_goal_marker(), create_robot(), create_static_wall(), drive_robot(), main() (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.18
Nodes (10): 1. Heatmap Figures, 2. Decision Outputs, 3. Deadlock Examples, Configurable Scenario, Current Scenario, Environment, NAMO Deadlock Prototype, Results Section (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.31
Nodes (9): build_world(), cell_center(), collect_contacts(), create_box(), create_cylinder(), drive_robot(), load_config(), run() (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.33
Nodes (5): generate_layout(), Returns (grid, start, goal, obstacle) for a given size., run(), Visualizer, run_evaluation()

### Community 9 - "Community 9"
Cohesion: 0.36
Nodes (4): NAMOPlanner, Calculates bypass cost C_by and removal cost C_re, then chooses the optimal stra, Standard A* pathfinder.         Returns the path length (cost) and the path coor, run_pipeline_demo()

### Community 10 - "Community 10"
Cohesion: 0.29
Nodes (7): render, boundary_color, box_color, camera_distance, robot_a_color, robot_b_color, wall_color

### Community 12 - "Community 12"
Cohesion: 0.40
Nodes (4): generate_deadlock_map(), print_grid_ascii(), Generates a hardcoded occupancy grid map of shape (grid_size, grid_size)     rep, Prints an ASCII visualization of the 2D grid.

## Knowledge Gaps
- **42 isolated node(s):** `name`, `grid_size`, `cell_size`, `wall_height`, `wall_thickness` (+37 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UNetDecisionPipeline` connect `Community 2` to `Community 1`, `Community 3`, `Community 4`, `Community 5`, `Community 8`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `Path` connect `Community 2` to `Community 1`, `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 8`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `NAMOmappoEnv` connect `Community 1` to `Community 2`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `UNetDecisionPipeline` (e.g. with `IntersectionEnvironment` and `WarehouseEnvironment`) actually correct?**
  _`UNetDecisionPipeline` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `NAMOmappoEnv` (e.g. with `ActorNetwork` and `CriticNetwork`) actually correct?**
  _`NAMOmappoEnv` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `NAMOEnvironment` (e.g. with `NAMOmappoEnv` and `NAMO3DSimulator`) actually correct?**
  _`NAMOEnvironment` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `WarehouseEnvironment` (e.g. with `UNetDecisionPipeline` and `NAMOmappoEnv`) actually correct?**
  _`WarehouseEnvironment` has 3 INFERRED edges - model-reasoned connections that need verification._