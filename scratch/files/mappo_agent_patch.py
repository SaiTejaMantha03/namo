"""
mappo_agent_patch.py
====================
NOT a standalone file. Surgical changes to mappo/mappo_agent.py.

Two methods change:
  1. ActorNetwork.forward()  — apply mask at logits stage (before softmax)
  2. MAPPOAgent.select_action() — accept and pass through the mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ════════════════════════════════════════════════════════════════════════════
# CHANGE 1 — ActorNetwork.forward()
# Replace your existing forward() with this version.
# The ONLY addition is the action_mask parameter and the two lines that
# apply it before softmax.
# ════════════════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
    """
    Paste this forward() into your existing ActorNetwork class.
    Keep your existing __init__ and any other methods unchanged.
    """

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.distributions.Distribution:
        """
        obs:         (batch, obs_dim) float tensor
        action_mask: (batch, action_dim) bool tensor, True = allowed.
                     None means all actions are allowed.

        Returns a Categorical distribution over actions.

        CRITICAL: mask is applied to LOGITS, not probabilities.
        Setting logits[~mask] = -inf ensures:
          (a) masked actions have 0 probability after softmax
          (b) gradient still flows through the unmasked logits
          (c) the network learns "I wanted X but couldn't" — not "X doesn't exist"
        """
        logits = self.mlp(obs)   # your existing forward logic here

        if action_mask is not None:
            # action_mask: True = allowed, False = blocked
            # We need to set blocked logits to -inf BEFORE softmax.
            # Using clone() to avoid in-place autograd issues.
            INF = torch.finfo(logits.dtype).max
            logits = logits.masked_fill(~action_mask, -INF)

        return Categorical(logits=logits)


# ════════════════════════════════════════════════════════════════════════════
# CHANGE 2 — MAPPOAgent.select_action()
# Replace your existing select_action() with this version.
# ════════════════════════════════════════════════════════════════════════════

class MAPPOAgent:
    """
    Paste this select_action() into your existing MAPPOAgent class.
    Keep your existing __init__, update(), store_transition(), save(), load().
    """

    def select_action(
        self,
        obs: torch.Tensor,
        action_mask: list[list[bool]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs:         (n_robots, obs_dim) float tensor or numpy array
        action_mask: list of n_robots boolean lists, each length action_dim.
                     e.g. [[True, False, True, True], [True, True, True, True]]
                     None means all actions allowed for all robots.

        Returns:
            actions   (n_robots,)     int64
            log_probs (n_robots,)     float32
            values    (n_robots,)     float32
        """
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs).to(self.device)

        # Convert action_mask list → tensor
        mask_tensor = None
        if action_mask is not None:
            mask_tensor = torch.tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            )   # shape: (n_robots, action_dim)

        with torch.no_grad():
            dist   = self.actor(obs, action_mask=mask_tensor)
            values = self.critic(obs).squeeze(-1)

        actions   = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs, values


# ════════════════════════════════════════════════════════════════════════════
# SANITY CHECK — run this before training to verify masking works
# ════════════════════════════════════════════════════════════════════════════

def test_masking():
    """
    Quick unit test. Run with: python mappo_agent_patch.py
    Expected output:
      PUSH (action 1) should have probability 0.0 when masked.
      WAIT (action 3) should have probability ~0.33 when unmasked.
    """
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    # Minimal actor for testing
    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(4, 4)

        def forward(self, obs, action_mask=None):
            logits = self.mlp(obs)
            if action_mask is not None:
                INF = torch.finfo(logits.dtype).max
                logits = logits.masked_fill(~action_mask, -INF)
            return Categorical(logits=logits)

    actor = _Actor()

    # 2 robots, 4 actions each
    obs = torch.randn(2, 4)

    # Robot 0: PUSH (idx 1) masked. Robot 1: all open.
    mask = torch.tensor([
        [True, False, True, True],   # robot 0: no PUSH
        [True, True,  True, True],   # robot 1: all ok
    ])

    dist   = actor(obs, action_mask=mask)
    probs  = dist.probs.detach()

    print("Robot 0 probs:", probs[0].tolist())
    print("Robot 1 probs:", probs[1].tolist())

    assert probs[0, 1].item() == 0.0, "PUSH should be 0 for robot 0"
    assert probs[1, 1].item() > 0.0, "PUSH should be available for robot 1"
    print("✓ Masking is correct.")


if __name__ == "__main__":
    test_masking()
