"""Generate simulated UAV localization examples from Esri imagery.

Workflow per case (matches the assignment's allowed simulation):
1. Choose true lat/lon and a shifted prior.
2. Fetch the search map around the PRIOR (same raster the service will see).
3. Crop / rotate / perturb a query patch at the TRUE location on that map.
4. Run the pipeline and save artifacts under examples/.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.georeferencer import METERS_PER_DEG_LAT, Georeferencer  # noqa: E402
from app.localizer import Localizer  # noqa: E402
from app.map_fetcher import MapFetcher  # noqa: E402
from app.models import LocalizationRequest  # noqa: E402
from app.pipeline import LocalizationPipeline  # noqa: E402
from app.visualize import make_visualization  # noqa: E402


@dataclass
class CaseSpec:
    name: str
    true_lat: float
    true_lon: float
    altitude_m: float
    prior_error_m: float
    prior_error_bearing_deg: float
    yaw_deg: float
    query_size: int
    hfov_deg: float
    search_radius_m: float
    brightness: float = 1.0
    contrast: float = 1.0
    noise_sigma: float = 4.0
    blur_ksize: int = 0
    color_shift: tuple[int, int, int] = (0, 0, 0)
    expect_success: bool = True


CASES = [
    CaseSpec(
        name="case_01_100m",
        true_lat=55.751244,
        true_lon=37.618423,
        altitude_m=100,
        prior_error_m=500,
        prior_error_bearing_deg=35,
        yaw_deg=15,
        query_size=512,
        hfov_deg=60,
        search_radius_m=900,
        noise_sigma=3,
    ),
    CaseSpec(
        name="case_02_300m",
        true_lat=55.703,
        true_lon=37.530,
        altitude_m=300,
        prior_error_m=2000,
        prior_error_bearing_deg=120,
        yaw_deg=-25,
        query_size=640,
        hfov_deg=65,
        search_radius_m=2800,
        noise_sigma=5,
        brightness=0.95,
    ),
    CaseSpec(
        name="case_03_600m",
        true_lat=59.931,
        true_lon=30.360,
        altitude_m=600,
        prior_error_m=5000,
        prior_error_bearing_deg=210,
        yaw_deg=40,
        query_size=640,
        hfov_deg=70,
        search_radius_m=6000,
        noise_sigma=6,
        contrast=0.9,
    ),
    CaseSpec(
        name="case_04_800m",
        true_lat=55.822,
        true_lon=37.640,
        altitude_m=800,
        prior_error_m=1500,
        prior_error_bearing_deg=80,
        yaw_deg=5,
        query_size=640,
        hfov_deg=60,
        search_radius_m=2500,
        brightness=0.25,
        contrast=1.8,
        noise_sigma=28,
        blur_ksize=11,
        color_shift=(70, -50, 25),
        expect_success=False,
    ),
]


def shift_latlon(
    lat: float, lon: float, distance_m: float, bearing_deg: float
) -> tuple[float, float]:
    bearing = math.radians(bearing_deg)
    dlat = (distance_m * math.cos(bearing)) / METERS_PER_DEG_LAT
    dlon = (distance_m * math.sin(bearing)) / (
        METERS_PER_DEG_LAT * max(math.cos(math.radians(lat)), 1e-6)
    )
    return lat + dlat, lon + dlon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def crop_query_from_map(
    map_bgr: np.ndarray,
    map_raster,
    true_lat: float,
    true_lon: float,
    case: CaseSpec,
) -> tuple[np.ndarray, float, float]:
    cx, cy = map_raster.latlon_to_pixel(true_lat, true_lon)
    ground_w = 2 * case.altitude_m * math.tan(math.radians(case.hfov_deg) / 2)
    m_per_px_x = (
        abs(map_raster.east - map_raster.west)
        * METERS_PER_DEG_LAT
        * max(math.cos(math.radians(true_lat)), 1e-6)
        / max(map_raster.width, 1)
    )
    side_px = int(
        max(64, min(map_raster.width, map_raster.height) * 0.35, ground_w / max(m_per_px_x, 1e-6))
    )
    # Keep query smaller than map for template matching.
    side_px = int(min(side_px, map_raster.width * 0.45, map_raster.height * 0.45))
    half = side_px // 2
    x0 = int(np.clip(cx - half, 0, max(0, map_raster.width - side_px)))
    y0 = int(np.clip(cy - half, 0, max(0, map_raster.height - side_px)))
    patch = map_bgr[y0 : y0 + side_px, x0 : x0 + side_px].copy()
    if patch.size == 0 or patch.shape[0] < 32:
        raise RuntimeError("Failed to crop query patch from search map")

    # GT = actual crop center (differs from requested true_* if clipped).
    gt_lat, gt_lon = map_raster.pixel_to_latlon(x0 + side_px / 2.0, y0 + side_px / 2.0)

    h, w = patch.shape[:2]
    if abs(case.yaw_deg) > 1e-3:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), case.yaw_deg, 1.0)
        patch = cv2.warpAffine(patch, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    crop = int(min(h, w) * 0.85)
    y1 = (h - crop) // 2
    x1 = (w - crop) // 2
    patch = patch[y1 : y1 + crop, x1 : x1 + crop]
    query = cv2.resize(
        patch, (case.query_size, case.query_size), interpolation=cv2.INTER_AREA
    )

    query_f = query.astype(np.float32)
    query_f = query_f * case.contrast + (case.brightness - 1.0) * 128.0
    query_f = query_f + np.array(case.color_shift, dtype=np.float32)
    if case.noise_sigma > 0:
        query_f = query_f + np.random.default_rng(0).normal(
            0, case.noise_sigma, query_f.shape
        )
    query = np.clip(query_f, 0, 255).astype(np.uint8)
    if case.blur_ksize and case.blur_ksize >= 3:
        k = case.blur_ksize | 1
        query = cv2.GaussianBlur(query, (k, k), 0)
    return query, gt_lat, gt_lon


def generate_case(case: CaseSpec, out_root: Path, fetcher: MapFetcher) -> dict:
    out_dir = out_root / case.name
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_lat, prior_lon = shift_latlon(
        case.true_lat, case.true_lon, case.prior_error_m, case.prior_error_bearing_deg
    )
    meta = {
        "lat_deg": prior_lat,
        "lon_deg": prior_lon,
        "altitude_m": case.altitude_m,
        "yaw_deg": case.yaw_deg,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "hfov_deg": case.hfov_deg,
        "search_radius_m": case.search_radius_m,
    }
    req = LocalizationRequest(**meta)

    # Fetch the same search map the service would use for this prior.
    plan = Georeferencer().plan(req, (case.query_size, case.query_size))
    search_map = fetcher.fetch(plan)

    query, gt_lat, gt_lon = crop_query_from_map(
        search_map.image, search_map, case.true_lat, case.true_lon, case
    )
    query_path = out_dir / "query.jpg"
    cv2.imwrite(str(query_path), query, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    expected = {
        "lat_deg": gt_lat,
        "lon_deg": gt_lon,
        "requested_true_lat_deg": case.true_lat,
        "requested_true_lon_deg": case.true_lon,
        "prior_error_m": case.prior_error_m,
        "expect_success": case.expect_success,
    }
    (out_dir / "expected.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )

    pipeline = LocalizationPipeline(
        map_fetcher=fetcher,
        localizer=Localizer(min_confidence=0.18, min_inliers=8),
    )
    ok = True
    error = None
    response = None
    try:
        response = pipeline.run(query_path.read_bytes(), req)
        result = response.model_dump()
        err_m = haversine_m(gt_lat, gt_lon, response.lat_deg, response.lon_deg)
        result["error_vs_expected_m"] = round(err_m, 2)
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(
            f"[OK] {case.name}: err={err_m:.1f} m, conf={response.confidence:.3f}, "
            f"t={response.processing_time_ms:.0f} ms"
        )
        try:
            vis = make_visualization(
                query,
                pipeline.last_map_raster or search_map,
                prior_lat,
                prior_lon,
                response.lat_deg,
                response.lon_deg,
            )
            cv2.imwrite(str(out_dir / "visualization.png"), vis)
        except Exception as viz_exc:  # noqa: BLE001
            print(f"[WARN] {case.name}: visualization failed: {viz_exc}")

        if case.expect_success and err_m > max(100.0, case.prior_error_m * 0.35):
            ok = False
            error = f"error too large: {err_m:.1f} m"
            print(f"[WEAK] {case.name}: {error}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        error = str(exc)
        (out_dir / "result.json").write_text(
            json.dumps(
                {"error": error, "expect_success": case.expect_success}, indent=2
            ),
            encoding="utf-8",
        )
        try:
            vis = make_visualization(
                query, search_map, prior_lat, prior_lon, None, None
            )
            cv2.imwrite(str(out_dir / "visualization.png"), vis)
        except Exception as viz_exc:  # noqa: BLE001
            print(f"[WARN] {case.name}: visualization failed: {viz_exc}")
        if case.expect_success:
            print(f"[FAIL] {case.name}: {error}")
        else:
            print(f"[EXPECTED FAIL] {case.name}: {error}")
            ok = True

    return {"name": case.name, "ok": ok, "error": error, "response": response}


def main() -> int:
    out_root = ROOT / "examples"
    out_root.mkdir(exist_ok=True)
    fetcher = MapFetcher(
        use_stub=False,
        cache_dir=ROOT / ".cache" / "tiles",
        max_tiles_per_side=18,
    )
    results = [generate_case(case, out_root, fetcher) for case in CASES]
    success = sum(1 for r in results if r["ok"])
    print(f"\nDone: {success}/{len(results)} cases acceptable")
    return 0 if success >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
