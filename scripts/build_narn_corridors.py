#!/usr/bin/env python3
"""Traverse the pulled NARN graph into named corridors, ready for `resample`.

This is the stage that was missing between `scripts/pull_narn.py` and the four
existing `routegen` stages. It is pure graph work -- no network access, no DEM --
so it is cheap and re-runnable, and every corridor it emits is a continuous
alignment with monotonic chainage that `resample_feature` accepts unchanged.

Output, under ``--out``:

``corridors/<corridor_id>.geojson``
    one GeoJSON LineString Feature per corridor, in the exact shape
    `routegen.io.centerline_to_geojson` produces, so the DEM and profile stages
    consume it without a special case.
``index.csv``
    one row per corridor: identity, owner, subdivision, division, states,
    length, vertex and segment counts, track class range.
``traversal_report.json``
    what the traversal did to the whole network -- group topology counts, chains
    grown, chains cut, chains dropped as too short, and the km in/out balance.

Run::

    python scripts/build_narn_corridors.py --narn data/narn --out data/narn_corridors
    python scripts/build_narn_corridors.py --state CO --out data/narn_corridors_co
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "route_generator"))

from routegen.io import centerline_to_geojson  # noqa: E402
from routegen.narn import (  # noqa: E402
    DEFAULT_MAX_LENGTH_M,
    DEFAULT_MIN_LENGTH_M,
    build_corridors,
    load_narn,
)

_INDEX_FIELDS = [
    "corridor_id", "owner", "subdiv", "division", "states", "country",
    "length_m", "num_vertices", "num_segments", "piece_index", "num_pieces",
    "was_cut", "fra_class_min", "fra_class_max", "tracks_max", "has_passenger",
    "first_arcid",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--narn", default="data/narn", help="directory written by pull_narn.py")
    ap.add_argument("--out", default="data/narn_corridors")
    ap.add_argument("--state", default=None, help="restrict to one STATEAB (for pilots)")
    ap.add_argument("--owner", default=None, help="restrict to one RROWNER1")
    ap.add_argument("--min-length-km", type=float, default=DEFAULT_MIN_LENGTH_M / 1000)
    ap.add_argument("--max-length-km", type=float, default=DEFAULT_MAX_LENGTH_M / 1000)
    ap.add_argument("--limit-groups", type=int, default=None)
    ap.add_argument("--no-geojson", action="store_true",
                    help="write the index and report only (fast dry run)")
    args = ap.parse_args()

    t0 = time.time()
    df, lon, lat = load_narn(args.narn)
    if args.state:
        df = df[df["STATEAB"] == args.state]
    if args.owner:
        df = df[df["RROWNER1"] == args.owner]
    print("loaded %s: %d segments, %.0f km" % (args.narn, len(df), df["KM"].sum()))

    corridors, report = build_corridors(
        df, lon, lat,
        min_length_m=args.min_length_km * 1000.0,
        max_length_m=args.max_length_km * 1000.0,
        limit_groups=args.limit_groups,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_geojson:
        gdir = out / "corridors"
        gdir.mkdir(exist_ok=True)
        for cl in corridors:
            cid = cl.relation_tags["corridor_id"]
            with open(gdir / ("%s.geojson" % cid), "w") as fh:
                json.dump(centerline_to_geojson(cl), fh)

    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_INDEX_FIELDS)
        w.writeheader()
        for m in report.corridors:
            w.writerow(asdict(m))

    rep = {k: v for k, v in report.__dict__.items() if k != "corridors"}
    rep["min_length_m"] = args.min_length_km * 1000.0
    rep["max_length_m"] = args.max_length_km * 1000.0
    rep["elapsed_s"] = round(time.time() - t0, 1)
    with open(out / "traversal_report.json", "w") as fh:
        json.dump(rep, fh, indent=2)

    print("\n--- traversal ---")
    for k, v in rep.items():
        print("  %-26s %s" % (k, round(v, 1) if isinstance(v, float) else v))
    lens = sorted(m.length_m / 1000.0 for m in report.corridors)
    if lens:
        mid = lens[len(lens) // 2]
        print("  corridor length km         min %.1f  median %.1f  max %.1f"
              % (lens[0], mid, lens[-1]))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
