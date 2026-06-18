# S-NAMO vs MAPPO Benchmarking Results

| Scenario | S-NAMO SR | S-NAMO Steps (Physics) | MAPPO SR | MAPPO Steps (Physics) | Winner | Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| namo_push_only | ✅ PASS | 81 | ✅ PASS | 90 | S-NAMO | S-NAMO (1.1x) |
| movable_obstacle_choke_namo | ✅ PASS | 200 | ✅ PASS | 240 | S-NAMO | S-NAMO (1.2x) |
| warehouse_small | ✅ PASS | 173 | ✅ PASS | 180 | S-NAMO | S-NAMO (1.0x) |
| warehouse_3robots | ✅ PASS | 373 | ✅ PASS | 360 | MAPPO | MAPPO (1.0x) |
| warehouse_large | ❌ FAIL | — | ❌ FAIL | — | None | Both Fail |
| single_corridor_yielding | ❌ FAIL | — | ❌ FAIL | — | None | Both Fail |
| symmetric_bottleneck_deadlock | ❌ FAIL | — | ❌ FAIL | — | None | Both Fail |
| narrow_doorway_congestion | ❌ FAIL | — | ❌ FAIL | — | None | Both Fail |
| cross_intersection | ❌ FAIL | — | ❌ FAIL | — | None | Both Fail |
| symmetric_bottleneck_4robots | ❌ FAIL | — | ❌ FAIL | — | None | Both Fail |
| custom_reconstructed_map_robots | ❌ FAIL | — | ❌ FAIL | — | None | Both Fail |