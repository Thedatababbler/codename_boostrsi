import json
import tempfile

from boosttopo.llm import CachedLLM, strip_thinking
from farm.adapter import ToyFarmAdapter, make_item
from farm.fake import FakeFarmLLM
from workflow.evaluate import Env

PROBLEM = {"days": 10, "cash": 10, "plots": 2,
           "crops": {"corn": {"cost": 5, "grow": 3, "price": 12},
                     "bean": {"cost": 2, "grow": 2, "price": 4}}}
ITEM = make_item(PROBLEM, "toy-0")

ILLEGAL = json.dumps([{"day": 1, "action": "plant", "crop": "corn", "plots": 3}])     # farmland
LEGAL_LOW = json.dumps([{"day": 1, "action": "plant", "crop": "bean", "plots": 1},
                        {"day": 3, "action": "harvest", "crop": "bean", "plots": 1}])

SKILL_OF = {"writer": "draft", "repairer": "repair", "improve": "improve", "analyst": "analyze",
            "critic": "critique", "y0": "y0"}


def ref_plan():
    from farm.reference import greedy_plan
    return json.dumps(greedy_plan(PROBLEM))


class ScriptedLLM:
    """Returns scripted plans per skill, in call order; records (skill, rep, prompt). y0 defaults to ILLEGAL.
    Used directly as env.llm (no cache), so every call reaches the script."""
    name = "scripted"
    temperature = 0.0

    def __init__(self, script: dict):
        self.script = {k: list(v) for k, v in script.items()}
        self.script.setdefault("y0", [ILLEGAL])
        self.calls = []

    def _next(self, skill):
        q = self.script.get(skill) or ["[]"]
        return q.pop(0) if len(q) > 1 else q[0]

    def call_y0(self, req):
        return f"```json\n{self._next('y0')}\n```", 50

    def call(self, req):
        skill = SKILL_OF[req.role]
        self.calls.append((skill, req.rep, req.prompt))
        return f"```json\n{self._next(skill)}\n```", 1100


def scripted_env(script):
    return Env(ScriptedLLM(script), ToyFarmAdapter(), workers=1, parallel_fanout=False)


def fake_env(cache_dir=None, **kw):
    cache_dir = cache_dir or tempfile.mkdtemp(prefix="wfcache_")
    return Env(CachedLLM(FakeFarmLLM(), cache_dir), ToyFarmAdapter(), workers=4, **kw)
