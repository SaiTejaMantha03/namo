# NAMO Benchmark Environments Layout

This document provides a visual walkthrough of all 9 benchmark environments defined in the configurations. These diagrams represent the starting configurations of the simulation grids, including fixed walls, movable boxes, robot spawn positions, and their target goals.

* **Grid Legend**:
  * **Charcoal Cells**: Impassable fixed walls.
  * **Orange Squares/Cells**: Movable obstacles (boxes).
  * **Light Grey Cells**: Free space.
  * **Colored Circles (e.g. R1, R2)**: Robot start positions.
  * **Colored Stars (e.g. G1, G2)**: Corresponding target goal positions.
  * **Dotted Lines**: Vector directions connecting each robot to its goal.

---

## Interactive Environment Carousel

````carousel
![Movable Obstacle Choke NAMO](/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/env_movable_obstacle_choke_namo.png)
### 1. Movable Obstacle Choke NAMO
* **Layout**: 15x15 grid with a single narrow corridor blocked by a movable box in the middle. 
* **Challenge**: Single robot (R1) must decide between a very long bypass detour or pushing the box (removal) to exit the corridor.
<!-- slide -->
![Warehouse Small](/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/env_warehouse_small.png)
### 2. Warehouse Small
* **Layout**: 10x10 grid containing narrow vertical warehouse aisles.
* **Challenge**: 2 robots moving in parallel. R1 is blocked by an orange box in its aisle, while R2 can move freely. R1 must push the box into a side-pocket to proceed.
<!-- slide -->
![Warehouse 3 Robots](/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/env_warehouse_3robots.png)
### 3. Warehouse 3 Robots
* **Layout**: 15x15 vertical multi-lane warehouse config.
* **Challenge**: 3 robots (R1, R2, R3) must navigate down three parallel lanes, each blocked by a box. Requires coordinated pushing behavior.
<!-- slide -->
![Single Corridor Yielding](/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/env_single_corridor_yielding.png)
### 4. Single Corridor Yielding
* **Layout**: 15x15 grid with a single-lane narrow hallway. 
* **Challenge**: 2 robots (R1, R2) enter from opposite sides, meeting head-on. The only way to resolve is for one robot to pull into the single side pocket (the yield pocket) to let the other pass.
<!-- slide -->
![Symmetric Bottleneck Deadlock](/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/env_symmetric_bottleneck_deadlock.png)
### 5. Symmetric Bottleneck Deadlock
* **Layout**: 15x15 bottleneck layout.
* **Challenge**: 2 robots enter a central narrow corridor bottleneck. Similar to single corridor yielding, but with symmetric entry routes forcing proactive yielding decisions.
<!-- slide -->
![Cross Intersection Coordination](/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/env_cross_intersection_coordination.png)
### 6. Cross Intersection Coordination
* **Layout**: 15x15 grid forming a tight central 4-way intersection.
* **Challenge**: 4 robots (R1, R2, R3, R4) converge on the center intersection from 4 directions. They must break symmetry, yield in sequence, and pass one by one without colliding.
<!-- slide -->
![Warehouse Large](/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/env_warehouse_large.png)
### 7. Warehouse Large
* **Layout**: Large 20x20 vertical aisle.
* **Challenge**: 2 robots must swap ends in a single vertical lane. The lane is blocked in the center by a box, and there is a side-pocket. The robots must push the box out of the way or pull over.
<!-- slide -->
![Narrow Doorway Congestion](/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/env_narrow_doorway_congestion.png)
### 8. Narrow Doorway Congestion
* **Layout**: 15x15 grid with a narrow doorway separating two large rooms.
* **Challenge**: 3 robots starting in the same room must navigate through a single narrow doorway to reach their goals. Highly congested traffic bottleneck.
<!-- slide -->
![Symmetric Bottleneck 4 Robots](/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/env_symmetric_bottleneck_4robots.png)
### 9. Symmetric Bottleneck 4 Robots
* **Layout**: 15x15 corridor with a central bottleneck.
* **Challenge**: 4 robots (2 on each side) must swap sides through a bottleneck corridor. High risk of gridlock requiring coordinated sequencing.
````
