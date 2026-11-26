"""Skills = fixed prompt templates (doc §3.1, §3.4).

Every skill renders the same template whether or not tools ran: tool outputs occupy fixed slots
that are left empty when the tool did not run, so with-tool / without-tool comparisons are fair.

NOTE: draft / repair / analyze are meant to be the *unchanged* writer / repairer / analyst prompts
from the original repo. That repo was not available here, so these are stand-ins; replace the
TEMPLATES entries verbatim with the originals (see docs/IMPLEMENTATION_NOTES.md).
"""
from __future__ import annotations

import json
import re

RULES = """Farm rules: days 1..D. Each plot holds one crop. `plant` costs cost*n cash now and needs n idle
plots; a crop planted on day d can be harvested on any day >= d+grow (and <= D), paying price*n and
freeing the plots. Unharvested crops are worth nothing. Goal: maximise final cash - initial cash.
Plan format: a JSON list of actions, e.g.
[{"day": 1, "action": "plant", "crop": "corn", "plots": 2},
 {"day": 5, "action": "harvest", "crop": "corn", "plots": 2}]"""

_SLOTS = """<crop_stats>
{crops}
</crop_stats>
<simulation>
{sim}
</simulation>
<analysis>
{analysis}
</analysis>"""

TEMPLATES = {
    "draft": """### SKILL: draft
You are an expert farm planner. Write a complete plan from scratch.
{rules}
<problem>
{problem}
</problem>
""" + _SLOTS + """
Think carefully, then output the final plan as one JSON list in a ```json block.""",

    "repair": """### SKILL: repair
The plan below is illegal. Fix it so it becomes legal, changing as little as possible.
{rules}
<problem>
{problem}
</problem>
<plan>
{plan}
</plan>
<diagnosis>
{diag}
</diagnosis>
Output the full corrected plan as one JSON list in a ```json block.""",

    "improve": """### SKILL: improve
The plan below is already legal. Increase its final profit while keeping it legal.
Use the simulation: days with idle land and spare cash are lost profit — fill them with crops that
can still mature before day D; prefer crops with higher profit per day; harvest as soon as crops
mature so the cash and land can be reused. Do not break legality: check cash and idle plots on
every day you change.
{rules}
<problem>
{problem}
</problem>
<plan>
{plan}
</plan>
""" + _SLOTS + """
Output the full improved plan as one JSON list in a ```json block.""",

    "analyze": """### SKILL: analyze
Write a short strategy analysis (<= 150 words) for this farm problem: which crops to prefer, when to
plant and harvest, cash bottlenecks. Do not write a plan.
{rules}
<problem>
{problem}
</problem>
""" + _SLOTS,
}


def render(skill: str, st) -> str:
    n = st.notes
    return TEMPLATES[skill].format(
        rules=RULES, problem=st.problem, plan=st.plan or "", diag=st.diag or "",
        crops=n.get("crops", ""), sim=n.get("sim", ""), analysis=n.get("analysis", ""))


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_output(skill: str, text: str) -> str:
    """Pull the plan (JSON list) out of a model response; analysis returns the text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if skill == "analyze":
        return text[:1200]
    blocks = _FENCE.findall(text)
    for b in reversed(blocks):
        try:
            v = json.loads(b.strip())
            if isinstance(v, list):
                return json.dumps(v)
        except Exception:  # noqa: BLE001
            pass
    a, b = text.rfind("["), text.rfind("]")
    # fall back to the last bracketed list
    start = text.rfind("[{")
    if start >= 0 and b > start:
        try:
            return json.dumps(json.loads(text[start:b + 1]))
        except Exception:  # noqa: BLE001
            pass
    return text[a:b + 1] if 0 <= a < b else text
