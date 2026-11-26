"""WorkflowProposer (doc §5.3): three candidate slots per step —
1 fragment, 1 control-structure atomic edit, 1 other atomic edit — each the policy's top choice."""
from __future__ import annotations

from dataclasses import dataclass, field

from .edits import legal_edits
from .model import Limits


@dataclass
class WorkflowProposer:
    limits: Limits = field(default_factory=Limits)
    allowed_steps: set | None = None       # None = all; the no-tools ablation removes tools
    allowed_fragments: set | None = None
    tried: set = field(default_factory=set)  # (frag_str, edit_str) pairs already evaluated

    def propose(self, frag, policy, obs: dict, context=None) -> list:
        from .model import to_str
        cands = [e for e in legal_edits(frag, self.limits, context, self.allowed_steps,
                                        self.allowed_fragments)
                 if (to_str(frag), str(e)) not in self.tried]
        slots = {"fragment": [], "control": [], "other": []}
        for e in cands:
            slots[e.category].append(e)
        out = []
        for name in ("fragment", "control", "other"):
            ranked = policy.rank(frag, slots[name], obs)
            if ranked:
                out.append(ranked[0])
        for e in out:
            self.tried.add((to_str(frag), str(e)))
        return out
