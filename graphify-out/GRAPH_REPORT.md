# Graph Report - New project  (2026-06-15)

## Corpus Check
- 48 files · ~43,079 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 407 nodes · 767 edges · 27 communities (24 shown, 3 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 76 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `569de373`
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
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]

## God Nodes (most connected - your core abstractions)
1. `ManipulationBeliefModel` - 27 edges
2. `SocialCostmap` - 26 edges
3. `UNetDecisionPipeline` - 25 edges
4. `NAMOEnvironment` - 22 edges
5. `WarehouseEnvironment` - 21 edges
6. `NAMOmappoEnv` - 20 edges
7. `BeliefBroadcaster` - 19 edges
8. `TrajectoryRegressionModel` - 18 edges
9. `simulate_env()` - 17 edges
10. `Path` - 17 edges

## Surprising Connections (you probably didn't know these)
- `NAMOPlanner` --uses--> `ManipulationBeliefModel`  [INFERRED]
  decision/namo_decision_pipeline.py → uncertainty/action_uncertainty.py
- `ndarray` --uses--> `ManipulationBeliefModel`  [INFERRED]
  decision/namo_decision_pipeline.py → uncertainty/action_uncertainty.py
- `run()` --calls--> `Path`  [INFERRED]
  decision/namo_visualizer.py → simulation/test_env.py
- `SNAMOPlanner` --uses--> `TabooZoneManager`  [INFERRED]
  decision/snamo_planner.py → social/taboo_zones.py
- `SNAMOPlanner` --uses--> `ManipulationBeliefModel`  [INFERRED]
  decision/snamo_planner.py → uncertainty/action_uncertainty.py

## Import Cycles
- None detected.

## Communities (27 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (41): box, friction, half_extent, height, mass, start_cell, corridor, left_opening_x (+33 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (15): ActorNetwork, CriticNetwork, MAPPOAgent, run_test(), a_star_internal(), find_clearing_direction_multi(), NAMOmappoEnv, Generates 20x20 risk heatmap from current layout for a specific agent. (+7 more)

### Community 2 - "Community 2"
Cohesion: 0.32
Nodes (13): a_star_internal(), cell_center(), create_box(), create_robot(), create_static_wall(), drive_robot(), load_custom_map_yaml(), main() (+5 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (9): Dataset, NAMODataset, train(), AttentionGate, AttentionUNet, DoubleConv, Attention Gate to filter skip connection features.     g: gating signal (coarser, Attention U-Net architecture tailored for NAMO occupancy grids.     Input: (batc (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (26): Backward-compatible wrapper around core.planner.a_star., Simulates a Gaussian Linear Regressor that predicts navigation cost         with, Generic decision pipeline supporting any size grid:         - Automatically pads, run_all_evaluations(), UNetDecisionPipeline, run_evaluation(), generate_demos(), IntersectionEnvironment (+18 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (39): Enum, BeliefBroadcaster, multi_robot/belief_broadcaster.py ----------------------------------- Cooperativ, Returns the current mean SR estimate for each robot., Central broker for cooperative belief sharing across a robot fleet.      In a re, Called by a robot immediately after a manipulation attempt.          Updates the, Returns the current [p_lo, p_hi] SR interval for a robot., Returns the SR interval width (p_hi - p_lo) for a robot.          Used by SR-Wid (+31 more)

### Community 6 - "Community 6"
Cohesion: 0.18
Nodes (10): 1. Heatmap Figures, 2. Decision Outputs, 3. Deadlock Examples, Configurable Scenario, Current Scenario, Environment, NAMO Deadlock Prototype, Results Section (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.31
Nodes (9): build_world(), cell_center(), collect_contacts(), create_box(), create_cylinder(), drive_robot(), load_config(), run() (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.39
Nodes (4): generate_layout(), Returns (grid, start, goal, obstacle) for a given size., run(), Visualizer

### Community 9 - "Community 9"
Cohesion: 0.12
Nodes (14): NAMOPlanner, ndarray, NAMOUnc BYPASS vs REMOVE decision using Laplace criterion.          Returns, Full NAMOUnc planning: decide and return waypoints.          Parameters, NAMOUnc-grade planner: uncertainty-aware BYPASS vs REMOVE decisions.      Parame, Update the Beta SR model after a manipulation attempt., Compute bypass cost interval [lo, hi] using Gaussian trajectory model.         O, Compute removal cost interval [lo, hi] accounting for SR uncertainty.          C (+6 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (25): a_star(), find_clearing_direction(), ndarray, core/planner.py --------------- Canonical A* planner and clearing-direction find, Given a movable box, find a (clear_cell, approach_cell) pair such that:       -, Risk-aware A* planner on a 2-D integer occupancy grid.      Grid values:, decision/namo_decision_pipeline.py ------------------------------------ NAMOUnc, decision/snamo_planner.py -------------------------- S-NAMO (Social NAMO) Planne (+17 more)

### Community 12 - "Community 12"
Cohesion: 0.40
Nodes (4): generate_deadlock_map(), print_grid_ascii(), Generates a hardcoded occupancy grid map of shape (grid_size, grid_size)     rep, Prints an ASCII visualization of the 2D grid.

### Community 23 - "Community 23"
Cohesion: 0.11
Nodes (15): ndarray, Find the push direction whose destination (clear_cell) has the         lowest so, S-NAMO plan: decide BYPASS / REMOVE and return full waypoints.          Paramete, Social NAMO planner with uncertainty-aware BYPASS/REMOVE decisions.      Paramet, Call after each real push attempt to update the SR belief., Recompute social map when the grid changes (box moved)., SNAMOPlanner, ndarray (+7 more)

### Community 24 - "Community 24"
Cohesion: 0.13
Nodes (10): _point_in_polygon(), social/taboo_zones.py ----------------------- Taboo zone management — prevents r, Ray-casting point-in-polygon test., Manages a set of taboo polygons and answers whether a given cell     falls insid, Pre-compute the full set of blocked (col, row) cells., Return True if the cell falls inside any taboo zone., Return the full set of taboo-blocked cells., Dynamically add a new taboo zone and update the blocked set. (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.22
Nodes (14): build_clean_grid(), cell_center(), create_box(), create_goal_marker(), create_robot(), create_wall(), drive_robot(), main() (+6 more)

## Knowledge Gaps
- **43 isolated node(s):** `name`, `grid_size`, `cell_size`, `wall_height`, `wall_thickness` (+38 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ManipulationBeliefModel` connect `Community 10` to `Community 9`, `Community 4`, `Community 5`, `Community 23`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `UNetDecisionPipeline` connect `Community 4` to `Community 1`, `Community 2`, `Community 3`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Why does `NAMOEnvironment` connect `Community 4` to `Community 1`, `Community 2`, `Community 5`, `Community 25`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `ManipulationBeliefModel` (e.g. with `NAMOPlanner` and `ndarray`) actually correct?**
  _`ManipulationBeliefModel` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `SocialCostmap` (e.g. with `ndarray` and `SNAMOPlanner`) actually correct?**
  _`SocialCostmap` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `UNetDecisionPipeline` (e.g. with `IntersectionEnvironment` and `WarehouseEnvironment`) actually correct?**
  _`UNetDecisionPipeline` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `NAMOEnvironment` (e.g. with `NAMOmappoEnv` and `ndarray`) actually correct?**
  _`NAMOEnvironment` has 4 INFERRED edges - model-reasoned connections that need verification._