import json
import re

from farm.bench import FarmBench, Item
from workflow.evaluate import Env
from workflow.fake import FakeFarmLLM
from workflow.llm import LLM, CachedLLM, Usage

PROBLEM = {"days": 10, "cash": 10, "plots": 2,
           "crops": {"corn": {"cost": 5, "grow": 3, "price": 12},
                     "bean": {"cost": 2, "grow": 2, "price": 4}}}
ITEM = Item("toy-0", json.dumps(PROBLEM), y0_score=0.0)

ILLEGAL = json.dumps([{"day": 1, "action": "plant", "crop": "corn", "plots": 3}])     # farmland
LEGAL_LOW = json.dumps([{"day": 1, "action": "plant", "crop": "bean", "plots": 1},
                        {"day": 3, "action": "harvest", "crop": "bean", "plots": 1}])
LEGAL_REF = None  # filled lazily: the reference plan


def ref_plan():
    from farm.reference import greedy_plan
    return json.dumps(greedy_plan(PROBLEM))


class ScriptedLLM(LLM):
    """Returns scripted plans per skill, in call order; records (skill, seed, prompt)."""
    model = "scripted"

    def __init__(self, script: dict):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def generate(self, prompt, seed):
        skill = re.search(r"### SKILL: (\w+)", prompt).group(1)
        self.calls.append((skill, seed, prompt))
        q = self.script.get(skill) or ["[]"]
        out = q.pop(0) if len(q) > 1 else q[0]
        return f"```json\n{out}\n```", Usage(100, 1000)


def scripted_env(script):
    return Env(ScriptedLLM(script), FarmBench(), workers=1, parallel_fanout=False)


def fake_env(cache=None):
    return Env(CachedLLM(FakeFarmLLM(), cache), FarmBench(), workers=4)
