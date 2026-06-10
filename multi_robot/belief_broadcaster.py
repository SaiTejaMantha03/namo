"""
multi_robot/belief_broadcaster.py
-----------------------------------
Cooperative belief sharing for manipulation success rate across a robot fleet.

Phase 5 — Novel Contribution.

Each robot maintains its own ManipulationBeliefModel (Beta distribution).
When any robot attempts a manipulation:
  - It records the outcome locally.
  - It broadcasts the outcome to all other robots.
  - Every other robot merges the incremental evidence into its own model.

Result: the fleet converges on an accurate SR estimate N× faster than any
single robot operating alone (N = fleet size).
"""

from __future__ import annotations
from uncertainty.action_uncertainty import ManipulationBeliefModel


class BeliefBroadcaster:
    """
    Central broker for cooperative belief sharing across a robot fleet.

    In a real deployment this would communicate over a network; here it
    operates in-process for simulation.

    Parameters
    ----------
    robot_ids      : list of robot IDs to manage.
    alpha, beta    : initial Beta prior shared by all robots.
    obstacle_type  : obstacle category all robots are tracking.
    """

    def __init__(
        self,
        robot_ids: list[int],
        alpha: float = 9.0,
        beta: float = 1.0,
        obstacle_type: str = "generic",
    ):
        self.belief_models: dict[int, ManipulationBeliefModel] = {
            rid: ManipulationBeliefModel(alpha, beta, obstacle_type)
            for rid in robot_ids
        }
        self._history: list[dict] = []   # log of all broadcast events

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def broadcast_outcome(
        self,
        robot_id: int,
        success: bool,
        obstacle_type: str = "generic",
    ) -> None:
        """
        Called by a robot immediately after a manipulation attempt.

        Updates the robot's own model, then propagates incremental evidence
        to all other robots via their merge() method.
        """
        if robot_id not in self.belief_models:
            return

        # Update the broadcasting robot's own model
        self.belief_models[robot_id].observe(success)

        # Share incremental evidence with all other robots
        src_model = self.belief_models[robot_id]
        for other_id, model in self.belief_models.items():
            if other_id != robot_id:
                model.merge(src_model)

        self._history.append({
            "robot_id": robot_id,
            "success": success,
            "obstacle_type": obstacle_type,
        })

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_sr_interval(
        self, robot_id: int, n_sigma: float = 2.0
    ) -> tuple[float, float]:
        """Returns the current [p_lo, p_hi] SR interval for a robot."""
        return self.belief_models[robot_id].success_rate_interval(n_sigma)

    def get_sr_interval_width(
        self, robot_id: int, n_sigma: float = 2.0
    ) -> float:
        """
        Returns the SR interval width (p_hi - p_lo) for a robot.

        Used by SR-Width DR to determine which robot is most uncertain
        and should therefore yield in a deadlock.
        """
        return self.belief_models[robot_id].interval_width(n_sigma)

    def get_all_means(self) -> dict[int, float]:
        """Returns the current mean SR estimate for each robot."""
        return {rid: m.mean() for rid, m in self.belief_models.items()}

    def summary(self) -> str:
        lines = ["BeliefBroadcaster summary:"]
        for rid, model in self.belief_models.items():
            lo, hi = model.success_rate_interval()
            lines.append(
                f"  Robot {rid}: SR={model.mean():.3f} "
                f"interval=[{lo:.3f},{hi:.3f}] "
                f"width={model.interval_width():.3f} "
                f"n_obs={model._n_obs}"
            )
        lines.append(f"  Total broadcasts: {len(self._history)}")
        return "\n".join(lines)
