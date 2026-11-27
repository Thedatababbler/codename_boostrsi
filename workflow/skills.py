"""Skills = prompts (doc §3.1, revision §12).

draft / repair / analyze / critique are the multi-agent writer / repairer / analyst / critic roles, rendered by
boosttopo.roles.role_prompt unchanged (task text = the benchmark's build_prompt, ledger tail included). Only
`improve` is new: it is written here, in the same layout, with the same ledger tail.

State -> role inputs:
- current plan  -> y_cur (the role prompt's CURRENT PLAN);
- diag          -> the repairer's SIMULATOR VERDICT;
- notes.analysis / notes.critique -> upstream messages (ANALYST NOTES / CRITIC NOTES), in the order they were
  written. Readers (analyze, critique) get no upstream, as in the multi-agent executor.

Tool outputs (notes.crops, notes.sim) are added as extra sections only when the tool ran. When no tool ran, the
prompt is byte-identical to the multi-agent role prompt (needed for cache replay). Deviation from §3.4 (fixed,
possibly empty slots): an empty slot would change every original prompt. Recorded in the notes.
"""
from __future__ import annotations

from boosttopo.llm import Request
from boosttopo.roles import _TAIL_WRITER, _fmt_solution, role_prompt

ROLE_OF = {"draft": "writer", "repair": "repairer", "analyze": "analyst", "critique": "critic"}
NOTE_ROLE = {"analysis": "analyst", "critique": "critic"}
READER_SKILLS = ("analyze", "critique")
WRITER_SKILLS = ("draft", "repair", "improve")
# skill -> tool notes it shows, with section titles
TOOL_SECTIONS = {
    "draft": (("crops", "CROP STATS (tool output)"),),
    "improve": (("crops", "CROP STATS (tool output)"), ("sim", "SIMULATION OF CURRENT PLAN (tool output)")),
    "analyze": (("crops", "CROP STATS (tool output)"),),
    "critique": (("sim", "SIMULATION OF CURRENT PLAN (tool output)"),),
    "repair": (),
}

IMPROVE_HEAD = (
    "You are the PLAN IMPROVER for a farm plan.\n"
    "The current plan is legal. Write a complete new plan that keeps every rule (funds, farmland, seed inventory, "
    "maturity, sell only harvested fruit) and earns strictly more simulated profit. Use the simulation: days with "
    "free plots and spare cash are lost profit -- fill them with crops that can still mature before the last day; "
    "prefer crops with higher profit per plot-day; harvest and sell as soon as crops mature so the cash and plots "
    "are reused. Check cash, free plots, seed and fruit inventory on every day you change."
)


def upstream_of(st) -> list[tuple[str, str]]:
    return [(NOTE_ROLE[k], v) for k, v in st.notes.items() if k in NOTE_ROLE and v]


def _tool_text(skill: str, st) -> str:
    parts = []
    for key, title in TOOL_SECTIONS.get(skill, ()):
        v = st.notes.get(key)
        if v:
            parts += [f"\n\n=== {title} ===\n", v.strip()]
    return "".join(parts)


def request(skill: str, st, item, rep: int) -> Request:
    plan = st.plan or ""
    up = [] if skill in READER_SKILLS else upstream_of(st)
    extra = _tool_text(skill, st)
    if skill == "improve":
        parts = [IMPROVE_HEAD, "\n\n=== TASK ===\n", item.prompt, "\n=== CURRENT PLAN ===\n", _fmt_solution(plan)]
        for r, msg in up:
            parts += [f"\n\n=== {'ANALYST NOTES' if r == 'analyst' else 'CRITIC NOTES'} ===\n", (msg or "").strip()]
        parts += [extra, _TAIL_WRITER]
        return Request(prompt="".join(parts), role="improve", item=item, y_cur=plan, upstream=up, rep=rep)
    role = ROLE_OF[skill]
    diag = (st.diag or "") if skill == "repair" else None
    req = role_prompt(role, item, plan, up, diag, rep=rep)
    if extra:
        if req.prompt.endswith(_TAIL_WRITER):
            req.prompt = req.prompt[: -len(_TAIL_WRITER)] + extra + _TAIL_WRITER
        else:
            req.prompt += extra
    return req
