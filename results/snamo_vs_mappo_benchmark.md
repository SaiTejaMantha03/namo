# S-NAMO vs MAPPO Benchmarking Results

| Scenario | S-NAMO SR | S-NAMO Steps (Physics) | MAPPO SR | MAPPO Steps (Physics) | Winner | Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| namo_push_only | ✅ PASS | 81 | ✅ PASS | 240 | S-NAMO | S-NAMO (3.0x) |
| movable_obstacle_choke_namo | ✅ PASS | 200 | ✅ PASS | 240 | S-NAMO | S-NAMO (1.2x) |
| warehouse_small | ✅ PASS | 173 | ✅ PASS | 270 | S-NAMO | S-NAMO (1.6x) |
| warehouse_3robots | ✅ PASS | 183 | ✅ PASS | 570 | S-NAMO | S-NAMO (3.1x) |
| warehouse_large | ✅ PASS | 734 | ✅ PASS | 1470 | S-NAMO | S-NAMO (2.0x) |
| single_corridor_yielding | ✅ PASS | 306 | ✅ PASS | 3420 | S-NAMO | S-NAMO (11.2x) |
| symmetric_bottleneck_deadlock | ✅ PASS | 301 | ✅ PASS | 3630 | S-NAMO | S-NAMO (12.1x) |
| narrow_doorway_congestion | ✅ PASS | 588 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| cross_intersection | ✅ PASS | 465 | ✅ PASS | 7410 | S-NAMO | S-NAMO (15.9x) |
| symmetric_bottleneck_4robots | ✅ PASS | 447 | ❌ FAIL | — | S-NAMO | S-NAMO (MAPPO FAIL) |
| custom_reconstructed_map_robots | ❌ FAIL | — | ✅ PASS | 2130 | MAPPO | MAPPO (S-NAMO FAIL) |