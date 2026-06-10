"""
social/taboo_zones.py
-----------------------
Taboo zone management — prevents robots from placing obstacles in
human-designated or dynamically-inferred forbidden regions.

Zones are specified as lists of polygon vertices (col, row), which are
converted to a set of blocked grid cells using point-in-polygon testing.

Paper reference: S-NAMO (Paper 2), Section IV — Taboo Zone Constraints.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def _point_in_polygon(px: float, py: float, polygon: list[tuple]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class TabooZoneManager:
    """
    Manages a set of taboo polygons and answers whether a given cell
    falls inside any of them.

    Parameters
    ----------
    zones       : list of polygon vertex lists, e.g.:
                  [[(3,5),(5,5),(5,8),(3,8)], ...]
    grid_size   : number of cells per side (used to build blocked cell set).
    """

    def __init__(self, zones: list[list[tuple]], grid_size: int = 20):
        self.zones = [list(map(tuple, z)) for z in zones]
        self.grid_size = grid_size
        self._blocked: set[tuple] = self._precompute_blocked()

    def _precompute_blocked(self) -> set[tuple]:
        """Pre-compute the full set of blocked (col, row) cells."""
        blocked = set()
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                for zone in self.zones:
                    if _point_in_polygon(col + 0.5, row + 0.5, zone):
                        blocked.add((col, row))
                        break
        return blocked

    def blocks(self, cell: tuple[int, int]) -> bool:
        """Return True if the cell falls inside any taboo zone."""
        return tuple(cell) in self._blocked

    def blocked_cells(self) -> set[tuple]:
        """Return the full set of taboo-blocked cells."""
        return self._blocked.copy()

    def add_zone(self, polygon: list[tuple]) -> None:
        """Dynamically add a new taboo zone and update the blocked set."""
        polygon = list(map(tuple, polygon))
        self.zones.append(polygon)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if _point_in_polygon(col + 0.5, row + 0.5, polygon):
                    self._blocked.add((col, row))

    @classmethod
    def from_yaml_config(cls, config: dict, grid_size: int = 20) -> "TabooZoneManager":
        """
        Build a TabooZoneManager from a parsed YAML config dict.

        Expected format:
            social:
              taboo_zones:
                - [[3,5],[5,5],[5,8],[3,8]]
        """
        social_cfg = config.get("social", {})
        raw_zones = social_cfg.get("taboo_zones", [])
        zones = [[tuple(v) for v in z] for z in raw_zones]
        return cls(zones, grid_size)

    def __repr__(self) -> str:
        return f"TabooZoneManager({len(self.zones)} zones, {len(self._blocked)} blocked cells)"
