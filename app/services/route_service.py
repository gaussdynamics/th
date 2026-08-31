"""Route catalog + profile loading for the Route screen.

Pure stdlib (json + math only) so it stays unit-testable without numpy or Qt.

Each file in ``route_generator/saved_routes/`` is a single GeoJSON LineString
Feature whose ``properties`` carry per-vertex arrays (``s_m``, ``elevation_m``,
``grade_pct``, ``kappa``, ``v_max_ms``) and scalar metadata (``length_m``,
``num_vertices``, ``maxspeed_coverage``, ``name``). The simulator-facing tensor
(`.npz`) is a separate concern handled by ``simulator2.route``; for *previewing*
a route the self-contained GeoJSON has everything we need.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# app/services/route_service.py -> parents[2] == repo root (th/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Processed profiles (3D coords + elevation/grade/kappa/v_max arrays) live here,
# alongside the matching .npz the simulator loads. Raw 2D centerlines in
# saved_routes/ have no elevation curve, so they are not the preview source.
DEFAULT_ROUTES_DIR = _REPO_ROOT / "route_generator" / "route_profiles"


@dataclass
class RouteEntry:
    """A row in the route list."""

    name: str
    path: str
    has_npz: bool = False  # simulator-ready tensor sits next to the geojson


@dataclass
class RouteData:
    """Full parsed route: aligned per-vertex arrays + summary statistics."""

    name: str
    path: str

    # per-vertex series (plain lists; NaN marks missing samples)
    s_m: List[float]
    elevation_m: List[float]
    grade_pct: List[float]
    kappa_1pm: List[float]
    v_max_kph: List[float]

    # scalar summary
    length_m: float
    num_vertices: int
    elev_min_m: float
    elev_max_m: float
    elev_gain_m: float
    elev_loss_m: float
    max_grade_pct: float
    min_grade_pct: float
    max_abs_kappa_1pm: float
    min_radius_m: float
    v_max_min_kph: float
    v_max_max_kph: float
    maxspeed_coverage: float

    @property
    def length_km(self) -> float:
        return self.length_m / 1000.0

    @property
    def s_km(self) -> List[float]:
        return [v / 1000.0 for v in self.s_m]


def _f(x) -> float:
    """Coerce to float; non-numeric / None -> NaN."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _finite(xs: List[float]) -> List[float]:
    return [v for v in xs if v == v and v not in (math.inf, -math.inf)]


class RouteService:
    def __init__(self, routes_dir: Path | str = DEFAULT_ROUTES_DIR) -> None:
        self.routes_dir = Path(routes_dir)

    def list_routes(self) -> List[RouteEntry]:
        if not self.routes_dir.exists():
            return []
        entries = [
            RouteEntry(
                name=p.stem,
                path=str(p),
                has_npz=p.with_suffix(".npz").exists(),
            )
            for p in sorted(self.routes_dir.glob("*.geojson"))
        ]
        return entries

    def load(self, path: Path | str) -> RouteData:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as fh:
            gj = json.load(fh)

        feat = gj["features"][0] if gj.get("type") == "FeatureCollection" else gj
        props = feat.get("properties", {})

        s = [_f(v) for v in props.get("s_m", [])]
        elev_raw = props.get("elevation_m") or props.get("z_smooth_m") or []
        elev = [_f(v) for v in elev_raw]
        grade = [_f(v) for v in props.get("grade_pct", [])]
        kappa = [_f(v) for v in props.get("kappa", [])]
        vmax_kph = [_f(v) * 3.6 for v in props.get("v_max_ms", [])]

        # --- summary stats (NaN-aware) ---
        elev_ok = _finite(elev)
        elev_min = min(elev_ok) if elev_ok else float("nan")
        elev_max = max(elev_ok) if elev_ok else float("nan")

        gain = loss = 0.0
        for a, b in zip(elev, elev[1:]):
            if a == a and b == b:  # both finite
                d = b - a
                if d > 0:
                    gain += d
                else:
                    loss += -d

        grade_ok = _finite(grade)
        max_grade = max(grade_ok) if grade_ok else float("nan")
        min_grade = min(grade_ok) if grade_ok else float("nan")

        abs_k = [abs(v) for v in _finite(kappa)]
        max_abs_k = max(abs_k) if abs_k else 0.0
        min_radius = (1.0 / max_abs_k) if max_abs_k > 0 else float("inf")

        v_ok = _finite(vmax_kph)
        v_min = min(v_ok) if v_ok else float("nan")
        v_max = max(v_ok) if v_ok else float("nan")

        length_m = _f(props.get("length_m"))
        if length_m != length_m:  # NaN -> fall back to last chainage
            length_m = max(_finite(s), default=0.0)
        num_vertices = int(props.get("num_vertices") or len(s))
        coverage = _f(props.get("maxspeed_coverage"))
        if coverage != coverage:
            coverage = 0.0

        return RouteData(
            name=props.get("name") or p.stem,
            path=str(p),
            s_m=s,
            elevation_m=elev,
            grade_pct=grade,
            kappa_1pm=kappa,
            v_max_kph=vmax_kph,
            length_m=length_m,
            num_vertices=num_vertices,
            elev_min_m=elev_min,
            elev_max_m=elev_max,
            elev_gain_m=gain,
            elev_loss_m=loss,
            max_grade_pct=max_grade,
            min_grade_pct=min_grade,
            max_abs_kappa_1pm=max_abs_k,
            min_radius_m=min_radius,
            v_max_min_kph=v_min,
            v_max_max_kph=v_max,
            maxspeed_coverage=coverage,
        )
