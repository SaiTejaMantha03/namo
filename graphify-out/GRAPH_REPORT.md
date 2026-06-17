# Graph Report - New project  (2026-06-16)

## Corpus Check
- 49 files · ~46,420 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 431 nodes · 858 edges · 32 communities (29 shown, 3 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 92 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d308082a`
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
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]

## God Nodes (most connected - your core abstractions)
1. `ManipulationBeliefModel` - 32 edges
2. `BeliefBroadcaster` - 30 edges
3. `SocialCostmap` - 29 edges
4. `UNetDecisionPipeline` - 25 edges
5. `TrajectoryRegressionModel` - 23 edges
6. `NAMOmappoEnv` - 22 edges
7. `NAMOEnvironment` - 22 edges
8. `WarehouseEnvironment` - 21 edges
9. `DeadlockResolver` - 20 edges
10. `Path` - 18 edges

## Surprising Connections (you probably didn't know these)
- `NAMOPlanner` --uses--> `ManipulationBeliefModel`  [INFERRED]
  decision/namo_decision_pipeline.py → uncertainty/action_uncertainty.py
- `ndarray` --uses--> `ManipulationBeliefModel`  [INFERRED]
  decision/namo_decision_pipeline.py → uncertainty/action_uncertainty.py
- `run()` --calls--> `Path`  [INFERRED]
  decision/namo_visualizer.py → simulation/test_env.py
- `SNAMOPlanner` --uses--> `SocialCostmap`  [INFERRED]
  decision/snamo_planner.py → social/social_costmap.py
- `SNAMOPlanner` --uses--> `ManipulationBeliefModel`  [INFERRED]
  decision/snamo_planner.py → uncertainty/action_uncertainty.py

## Import Cycles
- None detected.

## Communities (32 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (41): box, friction, half_extent, height, mass, start_cell, corridor, left_opening_x (+33 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (16): ActorNetwork, CriticNetwork, MAPPOAgent, run_test(), a_star_internal(), find_clearing_direction_multi(), NAMOmappoEnv, Generates 5x5 local risk crop from the UNet risk map around the agent's current (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.32
Nodes (13): a_star_internal(), cell_center(), create_box(), create_robot(), create_static_wall(), drive_robot(), load_custom_map_yaml(), main() (+5 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (9): Dataset, NAMODataset, train(), AttentionGate, AttentionUNet, DoubleConv, Attention Gate to filter skip connection features.     g: gating signal (coarser, Attention U-Net architecture tailored for NAMO occupancy grids.     Input: (batc (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (24): Backward-compatible wrapper around core.planner.a_star., Simulates a Gaussian Linear Regressor that predicts navigation cost         with, Generic decision pipeline supporting any size grid:         - Automatically pads, run_all_evaluations(), UNetDecisionPipeline, run_evaluation(), generate_demos(), IntersectionEnvironment (+16 more)

### Community 5 - "Community 5"
Cohesion: 0.24
Nodes (10): Enum, Conflict, ConflictDetector, ConflictType, multi_robot/conflict_detection.py ----------------------------------- Detects th, Inspect all robot plans simultaneously to find conflicts.      Usage     -----, ndarray, multi_robot/coordinator.py ---------------------------- Per-robot coordination l (+2 more)

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
Cohesion: 0.06
Nodes (38): BeliefBroadcaster, a_star(), find_clearing_direction(), ndarray, core/planner.py --------------- Canonical A* planner and clearing-direction find, Given a movable box, find a (clear_cell, approach_cell) pair such that:       -, Risk-aware A* planner on a 2-D integer occupancy grid.      Grid values:, NAMOPlanner (+30 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (10): ManipulationBeliefModel, Width of the SR confidence interval — measures current uncertainty., Incorporate evidence observed by another robot.          Adds only the *incremen, Serialise for network broadcasting., Deserialise a broadcast message., Bayesian Beta-distribution tracker for manipulation success rate.      Beta(alph, Online update after a single manipulation attempt.          Call this immediatel, Expected (mean) success rate. (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.40
Nodes (4): generate_deadlock_map(), print_grid_ascii(), Generates a hardcoded occupancy grid map of shape (grid_size, grid_size)     rep, Prints an ASCII visualization of the 2D grid.

### Community 23 - "Community 23"
Cohesion: 0.09
Nodes (16): ndarray, Find the push direction whose destination (clear_cell) has the         lowest so, S-NAMO plan: decide BYPASS / REMOVE and return full waypoints.          Paramete, Social NAMO planner with uncertainty-aware BYPASS/REMOVE decisions.      Paramet, Call after each real push attempt to update the SR belief., Recompute social map when the grid changes (box moved)., SNAMOPlanner, _point_in_polygon() (+8 more)

### Community 24 - "Community 24"
Cohesion: 0.15
Nodes (13): DeadlockResolver, _find_evasion_target(), _l2_norm(), _max_dist_cell(), ndarray, multi_robot/deadlock_resolution.py ------------------------------------ Deadlock, Legacy alias kept for API compatibility., Resolves multi-robot deadlocks using one of three strategies.      Parameters (+5 more)

### Community 25 - "Community 25"
Cohesion: 0.20
Nodes (15): build_clean_grid(), cell_center(), create_box(), create_goal_marker(), create_robot(), create_wall(), drive_robot(), main() (+7 more)

### Community 27 - "Community 27"
Cohesion: 0.15
Nodes (7): BeliefBroadcaster, multi_robot/belief_broadcaster.py ----------------------------------- Cooperativ, Returns the current mean SR estimate for each robot., Central broker for cooperative belief sharing across a robot fleet.      In a re, Called by a robot immediately after a manipulation attempt.          Updates the, Returns the current [p_lo, p_hi] SR interval for a robot., Returns the SR interval width (p_hi - p_lo) for a robot.          Used by SR-Wid

### Community 28 - "Community 28"
Cohesion: 0.24
Nodes (6): decision/snamo_planner.py -------------------------- S-NAMO (Social NAMO) Planne, social/social_costmap.py -------------------------- Derives a [0, 1] social cost, Return the social cost [0, 1] for a given (col, row) cell., Pre-computed social cost map for a given static obstacle layout.      Parameters, SocialCostmap, social/taboo_zones.py ----------------------- Taboo zone management — prevents r

### Community 29 - "Community 29"
Cohesion: 0.24
Nodes (5): ndarray, Derives social cost directly from UNet blockage probability output.          Pas, The full (grid_size, grid_size) cost array., Minimum ray-cast length in 4 cardinal directions (stops at wall)., Euclidean distance from each free cell to the nearest wall/box.

### Community 30 - "Community 30"
Cohesion: 0.33
Nodes (8): build_grid_from_config(), load_config(), main(), ndarray, run_coordination_eval.py -------------------------- Evaluation harness for multi, Build the numpy occupancy grid from a YAML config., Run a single coordination episode. Returns result dict., run_episode()

### Community 31 - "Community 31"
Cohesion: 0.40
Nodes (4): h = clamp(ceil(mean_pairwise_Manhattan / 2), 5, 20)., Run one timestep for ALL robots simultaneously.         Returns next_cells: {rob, All mutable state for a single robot across timesteps., RobotState

## Knowledge Gaps
- **43 isolated node(s):** `name`, `grid_size`, `cell_size`, `wall_height`, `wall_thickness` (+38 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ManipulationBeliefModel` connect `Community 10` to `Community 4`, `Community 9`, `Community 23`, `Community 27`, `Community 28`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **Why does `BeliefBroadcaster` connect `Community 27` to `Community 1`, `Community 4`, `Community 5`, `Community 9`, `Community 10`, `Community 24`, `Community 25`, `Community 30`, `Community 31`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **Why does `UNetDecisionPipeline` connect `Community 4` to `Community 1`, `Community 2`, `Community 3`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `ManipulationBeliefModel` (e.g. with `BeliefBroadcaster` and `NAMOPlanner`) actually correct?**
  _`ManipulationBeliefModel` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `BeliefBroadcaster` (e.g. with `BeliefBroadcaster` and `NAMOmappoEnv`) actually correct?**
  _`BeliefBroadcaster` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `SocialCostmap` (e.g. with `ndarray` and `SNAMOPlanner`) actually correct?**
  _`SocialCostmap` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `UNetDecisionPipeline` (e.g. with `IntersectionEnvironment` and `WarehouseEnvironment`) actually correct?**
  _`UNetDecisionPipeline` has 7 INFERRED edges - model-reasoned connections that need verification._