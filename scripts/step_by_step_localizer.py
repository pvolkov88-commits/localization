"""Step-by-step localization demo with saved figures.

Usage (from repo root):
  python scripts/step_by_step_localizer.py
  python scripts/step_by_step_localizer.py --case case_02_300m

Needs network (or warm .cache/tiles) for Esri imagery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.georeferencer import Georeferencer  # noqa: E402
from app.localizer import Localizer  # noqa: E402
from app.map_fetcher import MapFetcher  # noqa: E402
from app.models import LocalizationRequest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case_01_100m")
    args = parser.parse_args()

    case_dir = ROOT / "examples" / args.case
    out = ROOT / "notebooks" / "artifacts" / args.case
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    query_bgr = cv2.imread(str(case_dir / "query.jpg"))
    if query_bgr is None:
        raise SystemExit(f"Cannot read {case_dir / 'query.jpg'}")

    h, w = query_bgr.shape[:2]
    req = LocalizationRequest(**meta)
    plan = Georeferencer().plan(req, (w, h))
    print(f"GSD≈{plan.gsd_m:.3f} m/px zoom={plan.map_zoom} yaw={plan.yaw_deg}")

    map_raster = MapFetcher(cache_dir=ROOT / ".cache" / "tiles").fetch(plan)
    map_bgr = map_raster.image
    cv2.imwrite(str(out / "00_map.jpg"), map_bgr)
    print(f"map {map_raster.width}x{map_raster.height}")

    loc = Localizer()
    query_rot, _ = loc._rotate_image(query_bgr, -plan.yaw_deg)
    cv2.imwrite(str(out / "01_query_rotated.jpg"), query_rot)

    map_work, map_scale = loc._downscale(map_bgr, loc.max_map_side)
    query_work, _ = loc._downscale(query_rot, min(loc.max_map_side, max(query_rot.shape[:2])))
    q_clahe = loc._preprocess(query_work)
    m_clahe = loc._preprocess(map_work)
    cv2.imwrite(str(out / "02_query_clahe.jpg"), q_clahe)
    cv2.imwrite(str(out / "02_map_clahe.jpg"), m_clahe)

    coarse = loc._coarse_locate(query_work, map_work)
    if coarse is None:
        raise SystemExit("coarse matching failed")
    cx, cy, cw, ch, coarse_score = coarse
    print(f"coarse_score={coarse_score:.3f} ROI=({cx},{cy},{cw},{ch})")

    if coarse_score >= 0.25:
        pad = int(0.4 * max(cw, ch))
        x0, y0 = max(0, cx - pad), max(0, cy - pad)
        x1 = min(map_work.shape[1], cx + cw + pad)
        y1 = min(map_work.shape[0], cy + ch + pad)
        roi = map_work[y0:y1, x0:x1]
    else:
        # Same as Localizer: weak coarse → fine over the whole map.
        x0, y0 = 0, 0
        x1, y1 = map_work.shape[1], map_work.shape[0]
        roi = map_work
        print("coarse weak → fine on full map")

    map_vis = map_work.copy()
    cv2.rectangle(map_vis, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
    cv2.rectangle(map_vis, (x0, y0), (x1, y1), (0, 165, 255), 2)
    cv2.imwrite(str(out / "03_coarse_roi.jpg"), map_vis)
    cv2.imwrite(str(out / "03_roi.jpg"), roi)

    q_f = loc._preprocess(query_work)
    r_f = loc._preprocess(roi)
    kp_q, des_q, backend = loc._detect_describe(q_f)
    kp_r, des_r, _ = loc._detect_describe(r_f)
    print(f"backend={backend} kp_q={len(kp_q)} kp_r={len(kp_r)}")

    matcher = cv2.BFMatcher(cv2.NORM_L2 if backend == "sift" else cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_q, des_r, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(f"lowe matches={len(good)}")

    kp_img = cv2.drawKeypoints(
        query_work, kp_q, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(str(out / "04_keypoints_query.jpg"), kp_img)

    # Zoomed crop around the strongest keypoint for readability.
    best = max(kp_q, key=lambda k: k.response)
    bx, by = int(best.pt[0]), int(best.pt[1])
    zw = zh = 140
    hq, wq = query_work.shape[:2]
    zx0 = int(np.clip(bx - zw // 2, 0, max(0, wq - zw)))
    zy0 = int(np.clip(by - zh // 2, 0, max(0, hq - zh)))
    zx1, zy1 = zx0 + zw, zy0 + zh
    overview = query_work.copy()
    cv2.rectangle(overview, (zx0, zy0), (zx1, zy1), (0, 255, 0), 2)
    scale = 4
    crop = query_work[zy0:zy1, zx0:zx1]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    local = []
    for kp in kp_q:
        x, y = kp.pt
        if zx0 <= x < zx1 and zy0 <= y < zy1:
            local.append(
                cv2.KeyPoint(
                    (x - zx0) * scale,
                    (y - zy0) * scale,
                    max(kp.size * scale, 8.0),
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id,
                )
            )
    zoom_vis = cv2.drawKeypoints(
        big,
        local,
        None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    for lk in local:
        cx, cy = int(round(lk.pt[0])), int(round(lk.pt[1]))
        rad = max(4, int(round(lk.size / 2)))
        cv2.circle(zoom_vis, (cx, cy), rad, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(zoom_vis, (cx, cy), 2, (0, 0, 255), -1, cv2.LINE_AA)
        if lk.angle >= 0:
            ang = np.deg2rad(lk.angle)
            ex = int(round(cx + rad * np.cos(ang)))
            ey = int(round(cy + rad * np.sin(ang)))
            cv2.arrowedLine(zoom_vis, (cx, cy), (ex, ey), (255, 128, 0), 2, tipLength=0.35)
    cv2.imwrite(str(out / "04b_keypoints_overview_box.jpg"), overview)
    cv2.imwrite(str(out / "04b_keypoints_zoom.jpg"), zoom_vis)
    print(f"zoom keypoints in window: {len(local)} (anchor response={best.response:.3f})")

    lowe_img = cv2.drawMatches(
        query_work,
        kp_q,
        roi,
        kp_r,
        good[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out / "04_matches_lowe.jpg"), lowe_img)

    src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=4000,
        confidence=0.99,
    )
    if M is None or mask is None:
        raise SystemExit("RANSAC failed")

    inlier_mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(good, inlier_mask) if keep]
    print(f"inliers={int(inlier_mask.sum())}/{len(good)}")

    inlier_img = cv2.drawMatches(
        query_work,
        kp_q,
        roi,
        kp_r,
        inlier_matches[:100],
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out / "05_inliers.jpg"), inlier_img)

    qcx, qcy = query_work.shape[1] / 2.0, query_work.shape[0] / 2.0
    mapped = cv2.transform(np.array([[[qcx, qcy]]], dtype=np.float32), M)[0, 0]
    mx = (mapped[0] + x0) / map_scale
    my = (mapped[1] + y0) / map_scale
    lat, lon = map_raster.pixel_to_latlon(mx, my)

    gt_lat, gt_lon = expected["lat_deg"], expected["lon_deg"]
    err_m = float(
        np.hypot(
            (lat - gt_lat) * 111_320.0,
            (lon - gt_lon) * 111_320.0 * np.cos(np.radians(gt_lat)),
        )
    )
    print(f"estimated {lat:.6f},{lon:.6f}  GT {gt_lat:.6f},{gt_lon:.6f}  err≈{err_m:.1f}m")

    full = map_bgr.copy()
    for pt, color, label in (
        (map_raster.latlon_to_pixel(req.lat_deg, req.lon_deg), (255, 128, 0), "prior"),
        (map_raster.latlon_to_pixel(lat, lon), (0, 255, 0), "est"),
        (map_raster.latlon_to_pixel(gt_lat, gt_lon), (0, 0, 255), "GT"),
    ):
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        # OpenCV 5 drawMarker is picky about Point types — use lines instead.
        cv2.line(full, (x - 12, y), (x + 12, y), color, 2, cv2.LINE_AA)
        cv2.line(full, (x, y - 12), (x, y + 12), color, 2, cv2.LINE_AA)
        cv2.putText(full, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "06_result_on_map.jpg"), full)

    result = loc.localize(query_bgr, map_raster, plan)
    print(f"Localizer confidence={result.confidence:.3f} method={result.diagnostics.get('method')}")
    print(f"saved figures → {out}")


if __name__ == "__main__":
    main()
