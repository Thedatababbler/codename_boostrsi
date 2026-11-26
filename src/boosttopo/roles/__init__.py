"""Fixed role library (doc section 5.1) and prompt rendering."""

from __future__ import annotations

from ..llm import Request

WRITERS = ("writer", "repairer")
READERS = ("critic", "analyst")
ROLES = WRITERS + READERS


def prompt_version() -> str:
    """Hash of every prompt template; part of the evaluation-cache key."""
    import hashlib
    return hashlib.sha256(repr((_HEAD, _TAIL_WRITER)).encode()).hexdigest()[:10]


def is_writer(role: str) -> bool:
    return role in WRITERS


_HEAD = {
    "writer": (
        "You are the PLAN WRITER in a small team improving a farm plan.\n"
        "Write a complete new plan that earns strictly more simulated profit than the current plan, "
        "using any notes from teammates. Keep every rule (funds, farmland, seed inventory, maturity, "
        "sell only harvested fruit). If the current plan is empty, write one from scratch."
    ),
    "repairer": (
        "You are the PLAN REPAIRER in a small team improving a farm plan.\n"
        "The simulator's verdict on the current plan is given below. If the plan is illegal, fix every "
        "rule violation with the smallest changes and keep the plan's intent. If it is legal, fix the "
        "waste the verdict reports (idle plots, unsold fruit, leftover seeds, plantings that cannot mature) "
        "so that profit increases. Output the complete corrected plan."
    ),
    "critic": (
        "You are the PLAN CRITIC in a small team improving a farm plan.\n"
        "Do NOT write a plan. Check the current plan day by day: track cash, free plots, seed and fruit "
        "inventory. List concretely (a) every rule violation, (b) where cash or plots sit idle, (c) crops "
        "that are planted but cannot be harvested and sold before the last day, (d) better crop choices. "
        "Be concise: at most 15 bullet points."
    ),
    "analyst": (
        "You are the QUANTITATIVE ANALYST in a small team planning a farm.\n"
        "Do NOT write a plan. For each available crop compute: profit per harvest per plot "
        "(yield*fruit_price - seed_price), profit per plot-day, cash needed per plot, and the latest "
        "plant day that still allows harvest within the duration. Then recommend a cash-growth strategy "
        "(which crops to start with given the initial money, when to switch, how to use the final days). "
        "Be concise."
    ),
}

_TAIL_WRITER = ("\n\nFirst write a brief plain-text ledger, one line per day you act on: cash at the start, harvests, "
                "sales, purchases, plantings, cash and free plots at the end -- and check every rule as you go. Then "
                "output the complete final plan as one compact single-line JSON object in a ```json block "
                "(omit empty operation lists). The JSON block must be the last thing in your reply.")


def _fmt_solution(y: str) -> str:
    y = (y or "").strip()
    return y if y else "(empty -- no plan yet)"


def role_prompt(role: str, item, y_cur: str, upstream: list[tuple[str, str]], diag: str | None,
                rep: int = 0) -> Request:
    parts = [_HEAD[role], "\n\n=== TASK ===\n", item.prompt]
    if role != "analyst":
        parts += ["\n=== CURRENT PLAN ===\n", _fmt_solution(y_cur)]
    if role == "repairer":
        parts += ["\n\n=== SIMULATOR VERDICT ON CURRENT PLAN ===\n", diag or "(none)"]
    for r, msg in upstream:
        label = {"critic": "CRITIC NOTES", "analyst": "ANALYST NOTES"}.get(r, f"ALTERNATIVE PLAN FROM {r.upper()}")
        parts += [f"\n\n=== {label} ===\n", (msg or "").strip()]
    if is_writer(role):
        parts.append(_TAIL_WRITER)
    return Request(prompt="".join(parts), role=role, item=item, y_cur=y_cur, upstream=list(upstream),
                   diag=diag, rep=rep)


def y0_request(item, rep: int = 0) -> Request:
    """The single-writer baseline: the benchmark's own prompt, unchanged."""
    return Request(prompt=item.prompt, role="y0", item=item, y_cur="", rep=rep)
