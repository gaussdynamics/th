#!/usr/bin/env python3
"""Pull the North American Rail Network main-line graph from USDOT/BTS.

Replaces the Overpass crawl as the primary source of alignments. The reasons
are measured, not aesthetic:

topology is given, not inferred
    Every segment carries ``FRFRANODE``/``TOFRANODE``, so the rail graph is
    explicit. The OSM path crawled tile by tile and then reconstructed
    connectivity with stitching heuristics, which is where its gaps and
    branch-ordering problems came from.

the junk is already labelled
    ``NET`` classifies all 302,771 segments. ``M`` ("Main sub network") is
    95,936 of them; the remainder is 79k yard tracks, 88.5k minor industrial
    leads, passing sidings, out-of-service and abandoned line, and rail-trails.
    No heuristic filter needed.

it is not rate limited
    A Colorado query returned 1,307 segments and 41,402 vertices in 13.6 s.
    The Overpass crawl of the same region failed five consecutive seeds with
    HTTP 504 over 4.5 hours.

attributes the simulator and the thesis can use
    Owner, subdivision name, track count, passenger flag, mileage. Corridors
    get real identities ("MOFFAT TUNNEL") rather than ``route_line2``.

What NARN does *not* carry is line speed, which OSM ``maxspeed`` does. Elevation
still comes from 3DEP either way.

Output, under ``--out``:

``segments.parquet``
    one row per segment: FRAARCID, from/to node, owner, subdivision, state,
    track count, km, and the vertex slice into the geometry arrays.
``geometry.npz``
    ``lon``/``lat`` as one concatenated float64 array each, plus ``offsets``
    so segment ``i`` spans ``offsets[i]:offsets[i+1]``. Far smaller and faster
    to load than per-segment GeoJSON.

Run::

    python scripts/pull_narn.py --out data/narn
    python scripts/pull_narn.py --out data/narn_co --where "NET='M' AND STATEAB='CO'"
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE = ("https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/"
           "NTAD_North_American_Rail_Network_Lines/FeatureServer/0/query")

USER_AGENT = "thesis-routegen/0.1 (rail network acquisition; research)"

#: Kept per segment. The trackage-rights and multi-owner columns are dropped:
#: nine TRKRGHTS fields inflate the table and none of them affect the physics.
FIELDS = ("FRAARCID", "FRFRANODE", "TOFRANODE", "RROWNER1", "SUBDIV", "BRANCH",
          "DIVISION", "STATEAB", "COUNTRY", "TRACKS", "PASSNGR", "NET", "KM")


def _get(params: dict, timeout: float, retries: int = 4) -> dict:
    """POST rather than GET.

    An id list of a few hundred segments makes a URL long enough for the
    service to answer 404, which reads like a wrong endpoint rather than an
    oversized request. ArcGIS accepts the same parameters form-encoded.
    """
    body = urllib.parse.urlencode(params).encode("utf-8")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                SERVICE, data=body,
                headers={"User-Agent": USER_AGENT,
                         "Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"query failed after {retries} attempts: {last}")


def fetch_ids(where: str, timeout: float) -> np.ndarray:
    d = _get({"where": where, "returnIdsOnly": "true", "f": "json"}, timeout)
    ids = np.asarray(d.get("objectIds") or [], dtype=np.int64)
    if ids.size == 0:
        raise SystemExit(f"no features match {where!r}")
    return np.sort(ids)


def fetch_chunk(ids: np.ndarray, timeout: float) -> list[dict]:
    """Fetch by explicit object id, which pages reliably.

    ``resultOffset`` degrades badly at large offsets on this service; an
    explicit id list costs one URL per chunk and is order-stable.
    """
    d = _get({
        "objectIds": ",".join(str(i) for i in ids),
        "outFields": ",".join(FIELDS),
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
    }, timeout)
    return d.get("features", [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--where", default="NET='M'",
                    help="server-side filter (default: main sub network)")
    ap.add_argument("--chunk", type=int, default=400,
                    help="segments per request; the service caps at 2000 records "
                         "but long id lists make long URLs")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"where: {args.where}")

    t0 = time.time()
    ids = fetch_ids(args.where, args.timeout)
    print(f"{len(ids):,} segments to fetch, {len(ids)//args.chunk + 1} requests\n")

    rows: list[dict] = []
    lons: list[np.ndarray] = []
    lats: list[np.ndarray] = []
    offsets: list[int] = [0]
    n_multi = 0

    for c, start in enumerate(range(0, len(ids), args.chunk)):
        feats = fetch_chunk(ids[start:start + args.chunk], args.timeout)
        for f in feats:
            paths = f["geometry"].get("paths") or []
            if not paths:
                continue
            if len(paths) > 1:
                n_multi += 1
            # A multi-path segment is rare here; concatenating its parts keeps
            # one row per FRAARCID, and the node ids still carry the topology.
            pts = np.asarray([p for path in paths for p in path], dtype=np.float64)
            lons.append(pts[:, 0])
            lats.append(pts[:, 1])
            offsets.append(offsets[-1] + pts.shape[0])
            a = f["attributes"]
            rows.append({k: a.get(k) for k in FIELDS})

        done = min(start + args.chunk, len(ids))
        if c % 10 == 0 or done == len(ids):
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f"  {done:7,}/{len(ids):,}  {el:6.0f}s  "
                  f"{rate:6.0f} seg/s  eta {(len(ids)-done)/max(rate,1e-9):5.0f}s")

    df = pd.DataFrame(rows)
    df["vertex_start"] = offsets[:-1]
    df["vertex_end"] = offsets[1:]
    df.to_parquet(args.out / "segments.parquet", index=False)
    np.savez_compressed(
        args.out / "geometry.npz",
        lon=np.concatenate(lons), lat=np.concatenate(lats),
        offsets=np.asarray(offsets, dtype=np.int64),
    )

    km = float(df["KM"].fillna(0).sum())
    size = sum(p.stat().st_size for p in args.out.iterdir()) / 2 ** 20
    print(f"\nwrote {len(df):,} segments, {km:,.0f} km, "
          f"{offsets[-1]:,} vertices  ({size:.1f} MB) in {time.time()-t0:.0f}s")
    if n_multi:
        print(f"  {n_multi} multi-path segments concatenated")
    nodes = set(df.FRFRANODE) | set(df.TOFRANODE)
    print(f"  graph: {len(nodes):,} nodes, {len(df):,} edges")
    print(f"  countries: {df.COUNTRY.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
