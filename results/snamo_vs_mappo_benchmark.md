# S-NAMO vs MAPPO Benchmarking Results

| Scenario | S-NAMO SR | S-NAMO Steps (Physics) | MAPPO SR | MAPPO Steps (Physics) | Winner | Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| movable_obstacle_choke_namo | ✅ PASS | 199 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| warehouse_small | ✅ PASS | 173 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| warehouse_3robots | ✅ PASS | 183 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| warehouse_large | ✅ PASS | 749 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| single_corridor_yielding | ✅ PASS | 291 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| symmetric_bottleneck_deadlock | ✅ PASS | 311 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| narrow_doorway_congestion | ✅ PASS | 593 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| cross_intersection | ✅ PASS | 408 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| symmetric_bottleneck_4robots | ✅ PASS | 526 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| custom_reconstructed_map_robots | ❌ FAIL | — | ❌ FAIL | — | None | Both Fail |