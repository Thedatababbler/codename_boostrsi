"""All baselines and hand-written workflows, expressed as workflows so they run through the same
executor as the searched methods (doc §7.1, §7.2, §8.1, §9.1)."""
from __future__ import annotations

import math
import statistics

from .model import parse_wf, worst_case_calls


def single_draft():
    return parse_wf("seq[draft]")


def best_of(n: int):                       # B1
    return parse_wf(f"seq[fanout(n={n})[draft]]")


def best_of_with_tools(n: int):            # B2
    return parse_wf(f"seq[crop_stats, fanout(n={n})[draft]]")


def repair_loop(k: int):                   # B3
    return parse_wf(f"seq[draft, loop(k={k}, stop=best)[validate, repair]]")


HANDWRITTEN = {
    "F1_sim_improve": parse_wf("seq[draft, simulate, improve]"),
    "F2_stats_split": parse_wf("seq[crop_stats, draft, branch(status){"
                               "illegal: loop(k=2, stop=legal)[validate, repair], "
                               "legal_low: seq[simulate, improve]}]"),
    "F3_fanout_improve": parse_wf("seq[fanout(n=3)[draft], simulate, improve]"),
}


def precheck_methods() -> dict:
    """F1-F3 plus call-matched B1/B2/B3 (B3 has 1 draft + k repairs = calls)."""
    m = {"single_draft": single_draft()}
    m.update(HANDWRITTEN)
    for c in sorted({worst_case_calls(w) for w in HANDWRITTEN.values()}):
        m[f"B1_best_of_{c}"] = best_of(c)
        m[f"B2_tools_best_of_{c}"] = best_of_with_tools(c)
        m[f"B3_repair_loop_k{c - 1}"] = repair_loop(c - 1)
    return m


# Methods whose requests are the multi-agent ones: run with seed_scheme "legacy" so they are served from the
# multi-agent cache (single_draft = writer_once, B1 = best_of_n_writer, B3 = Stage A's repair loop). Everything
# else uses seed_scheme "path". diag_mode stays "clean" for all of them (a new comparison), so B3's repairs of
# legal plans (whose diagnosis contains the reference profit in the original) are new requests.
LEGACY_SEEDED = ("single_draft", "B1_", "B3_")


def seed_scheme_of(name: str) -> str:
    return "legacy" if name.startswith(LEGACY_SEEDED) else "path"


def token_aligned_n(our_tokens_per_item: float, draft_tokens_per_call: float) -> int:
    """§8.1: n rounded *down* so the baseline does not exceed our actual token spend."""
    if draft_tokens_per_call <= 0:
        return 1
    return max(1, math.floor(our_tokens_per_item / draft_tokens_per_call + 1e-9))


def token_aligned_k(our_tokens_per_item: float, draft_tokens: float, repair_tokens: float) -> int:
    if repair_tokens <= 0:
        return 1
    return max(1, math.floor((our_tokens_per_item - draft_tokens) / repair_tokens + 1e-9))


def final_baselines(our_results, single_results, repair_results=None, best_handwritten=None) -> dict:
    """Build the §8.1 baseline set, cost-aligned to our method's measured mean tokens/item."""
    ex = lambda rs: statistics.fmean(r.cost.tokens - r.cost.y0_tokens for r in rs)  # noqa: E731 (y0 is shared)
    ours, d = ex(our_results), ex(single_results)
    n = token_aligned_n(ours, d)
    out = {"single_draft": single_draft(), f"best_of_{n}": best_of(n),
           f"tools_best_of_{n}": best_of_with_tools(n)}
    if repair_results:
        rep = ex(repair_results) - d
        k = token_aligned_k(ours, d, max(rep, 1))
        out[f"repair_loop_k{k}"] = repair_loop(k)
    if best_handwritten is not None:
        out[best_handwritten[0]] = best_handwritten[1]
    return out
