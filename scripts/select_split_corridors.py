#!/usr/bin/env python3
"""Choose the corridor pool to elevate for the split set.

The splits need corridors of known grade, and grade is only known after
elevation, so this script cannot assign splits -- it chooses what to *measure*.
`assign_split_corridors.py` does the assignment afterwards, from measured
grade. Keeping the two apart matters: a corridor is called steep because its
grade field says so, never because its name or its state suggested it.

Two strata, because a uniform draw will not do:

**Representative.** An evenly spaced, deterministic walk through the US index.
This is the pool that train / val / test_id / ood_corridor come from, and it
has to carry the terrain mix of the real network, whatever that is.

**Mountain candidates.** A uniform draw is about 93% gentle -- 5 of 75 corridors
in the offline sample had rms grade above 1.25% -- so `ood_grade` would come up
empty. Corridors are flagged as candidates by subdivision name and by state,
both of which are cheap and available before any elevation is spent. The proxy
only decides who gets measured; if a flagged corridor turns out flat, the
assignment step will put it in the ordinary pool.

Canada and Mexico are excluded: 3DEP is a USGS product and the QA gate rejected
5 of 9 Canadian and 1 of 3 Mexican corridors against 0 of 48 US ones. The
network is a US network until a second DEM is sourced.

Run::

    python scripts/select_split_corridors.py --n-representative 200 --n-mountain 80
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Subdivision names that suggest grade. Matched case-insensitively as substrings
# of the subdiv field. Deliberately generous: a false positive costs one
# corridor's elevation, a false negative costs a corridor ood_grade may need.
MOUNTAIN_WORDS = (
    "pass", "tunnel", "canyon", "summit", "mountain", "divide", "grade",
    "hill", "gorge", "ridge", "range", "peak", "cascade", "sierra",
)

# States where North American mountain railroading actually happens. Used as a
# second, independent flag -- a corridor qualifies on either.
MOUNTAIN_STATES = frozenset(
    "CO WY MT UT ID NV CA OR WA NM AZ WV VA TN NC KY PA".split()
)


def matches_mountain_name(row: dict) -> bool:
    subdiv = (row.get("subdiv") or "").lower()
    return any(w in subdiv for w in MOUNTAIN_WORDS)


def matches_mountain_state(row: dict) -> bool:
    states = {s.strip().upper() for s in (row.get("states") or "").split(",") if s.strip()}
    return bool(states & MOUNTAIN_STATES)


def is_mountain_candidate(row: dict) -> bool:
    return matches_mountain_name(row) or matches_mountain_state(row)


def evenly_spaced(items: list, n: int) -> list:
    """Deterministic evenly spaced pick -- no RNG, so re-running is stable and
    the choice does not depend on a seed nobody recorded."""
    if n >= len(items):
        return list(items)
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corridors", type=Path, default=Path("data/narn_corridors"))
    ap.add_argument("--out", type=Path, default=Path("data/split_corridor_pool.json"))
    ap.add_argument("--n-representative", type=int, default=200)
    ap.add_argument("--n-mountain", type=int, default=200)
    ap.add_argument("--country", default="US")
    args = ap.parse_args()

    with open(args.corridors / "index.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["country"] == args.country]
    rows.sort(key=lambda r: r["corridor_id"])
    print(f"{len(rows)} {args.country} corridors in the index")

    # Name and state are not equally informative and must not be pooled. The
    # state flag hits 29% of the US network -- evenly spacing across that gives
    # back a representative sample, which is what it did on the first run, and
    # ood_grade came out with a +0.02 pp separation from training. The name flag
    # hits 3%, so every name match is taken and the state flag only tops up.
    by_name = [r for r in rows if matches_mountain_name(r)]
    by_state = [r for r in rows
                if not matches_mountain_name(r) and matches_mountain_state(r)]
    topup = evenly_spaced(by_state, max(0, args.n_mountain - len(by_name)))
    mountain = by_name + topup
    mountain_ids = {r["corridor_id"] for r in mountain}
    print(f"{len(by_name)} name matches (all taken); "
          f"{len(by_state)} state-only -> topping up with {len(topup)}")

    # The representative stratum is drawn from everything, mountain corridors
    # included, so it keeps the network's real terrain mix; ids already taken
    # are skipped rather than duplicated.
    rest = [r for r in rows if r["corridor_id"] not in mountain_ids]
    representative = evenly_spaced(rest, args.n_representative)
    print(f"{len(representative)} representative corridors")

    pool = {
        "country": args.country,
        "n_representative": len(representative),
        "n_mountain": len(mountain),
        "mountain_words": list(MOUNTAIN_WORDS),
        "mountain_states": sorted(MOUNTAIN_STATES),
        "representative": sorted(r["corridor_id"] for r in representative),
        "mountain_candidates": sorted(mountain_ids),
    }
    pool["all"] = sorted(set(pool["representative"]) | set(pool["mountain_candidates"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pool, indent=2))

    est_pts = sum(float(r["length_m"]) / 10.0 for r in representative + mountain)
    print(f"\n{len(pool['all'])} corridors, ~{est_pts/1000:.0f}k elevation points")
    print(f"  at 420 pts/s sustained: {est_pts/420/60:.0f} min")
    print(f"  at 900 pts/s:           {est_pts/900/60:.0f} min")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
