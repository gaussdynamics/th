#!/usr/bin/env python3
"""Headless bulk route-dataset builder (CLI).

Crawls a curated list of seed points into a deduplicated route dataset. Designed
to run in batches and resume: each seed's result is cached under ``raw/``, so you
can stop and re-run, or add seeds and only the new ones are crawled.

Examples
--------
    # First 5 seeds only, small crawl, to validate the pipeline end-to-end:
    python build_dataset.py --seeds seeds/seeds_us.json --out data --limit 5 \
        --seed-radius-km 25 --max-radius-km 120

    # Full national run on a permissive endpoint, polite 1.5s between seeds:
    python build_dataset.py --seeds seeds/seeds_us.json --out data \
        --endpoint https://overpass.kumi.systems/api/interpreter --sleep 1.5

    # Re-run only the dedup/write stage after editing raw/ (no network):
    python build_dataset.py --seeds seeds/seeds_us.json --out data --dedup-only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routegen.dataset import load_seeds, run_acquisition  # noqa: E402
from routegen.osm import DEFAULT_ENDPOINT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Bulk railway route dataset builder")
    ap.add_argument("--seeds", required=True, help="JSON seed file (list of {name,lat,lon,radius_km?})")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--seed-radius-km", type=float, default=30.0)
    ap.add_argument("--max-radius-km", type=float, default=250.0, help="Hard crawl distance cap")
    ap.add_argument("--expand-m", type=float, default=60.0)
    ap.add_argument("--max-iterations", type=int, default=40)
    ap.add_argument("--min-len-km", type=float, default=1.0)
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between seeds (be polite)")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N seeds")
    ap.add_argument("--seed-tile-km", type=float, default=8.0,
                    help="Seed pull is tiled into around-queries of this spacing (smaller = lighter)")
    ap.add_argument("--retries", type=int, default=3, help="Per-query retries on busy/timeout")
    ap.add_argument("--all-tracks", action="store_true",
                    help="Include yards/sidings/industrial spurs (default: main lines only)")
    ap.add_argument("--no-crawl", action="store_true", help="Radius-only (no outward crawl)")
    ap.add_argument("--no-resume", action="store_true", help="Re-crawl even if cached")
    ap.add_argument("--dedup-only", action="store_true", help="Skip crawling; just dedup+write")
    args = ap.parse_args()

    seeds = load_seeds(args.seeds)
    print(f"Loaded {len(seeds)} seeds from {args.seeds}")
    run_acquisition(
        seeds, args.out,
        seed_radius_km=args.seed_radius_km,
        crawl=not args.no_crawl,
        max_radius_km=args.max_radius_km,
        expand_m=args.expand_m,
        max_iterations=args.max_iterations,
        min_len_km=args.min_len_km,
        main_only=not args.all_tracks,
        seed_tile_km=args.seed_tile_km,
        retries=args.retries,
        endpoint=args.endpoint,
        sleep_s=args.sleep,
        limit=args.limit,
        resume=not args.no_resume,
        dedup_only=args.dedup_only,
    )


if __name__ == "__main__":
    main()
