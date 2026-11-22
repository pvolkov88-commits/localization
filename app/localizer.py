"""Localizer: coarse template match + fine feature matching (SIFT/ORB + RANSAC)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.exceptions import LocalizationNotConverged
from app.georeferencer import SearchPlan
from app.map_fetcher import MapRaster

logger = logging.getLogger(__name__)


@dataclass
class LocalizationResult:
    lat_deg: float
    lon_deg: float
    confidence: float
    diagnostics: dict


class Localizer:
    def __init__(
        self,
        echo_prior: bool = False,
        min_confidence: float = 0.25,
        min_inliers: int = 10,
        max_map_side: int = 2560,
        coarse_max_side: int = 1024,
    ) -> None:
        self.echo_prior = echo_prior
        self.min_confidence = min_confidence
        self.min_inliers = min_inliers
        self.max_map_side = max_map_side
        self.coarse_max_side = coarse_max_side

    def localize(
        self,
        query_bgr: np.ndarray,
        map_raster: MapRaster,
        plan: SearchPlan,
    ) -> LocalizationResult:
        if query_bgr is None or query_bgr.size == 0:
            raise LocalizationNotConverged("Empty query image")

        if self.echo_prior:
            return LocalizationResult(
                lat_deg=plan.center_lat_deg,
                lon_deg=plan.center_lon_deg,
                confidence=0.0,
                diagnostics={
                    "note": "echo_prior stub — no matching performed",
                    "map_source": map_raster.source,
                    "map_zoom": map_raster.zoom,
                    "gsd_m": plan.gsd_m,
                    "n_inliers": 0,
                },
            )

        map_bgr = map_raster.image
        if map_bgr.size == 0:
            raise LocalizationNotConverged("Empty map raster")

        # Align query roughly north-up using yaw prior (yaw=0 means already north-up).
        query_rot, rot_m = self._rotate_image(query_bgr, -plan.yaw_deg)

        map_work, map_scale = self._downscale(map_bgr, self.max_map_side)
        query_work, _ = self._downscale(query_rot, min(self.max_map_side, max(query_rot.shape[:2])))

        # --- Coarse: template matching on downscaled images ---
        coarse = self._coarse_locate(query_work, map_work)
        if coarse is None:
            raise LocalizationNotConverged("Coarse template matching failed")

        cx, cy, cw, ch, coarse_score = coarse
        # If coarse is weak, search the whole (scaled) map at the fine stage.
        if coarse_score >= 0.25:
            pad = int(0.4 * max(cw, ch))
            x0 = max(0, cx - pad)
            y0 = max(0, cy - pad)
            x1 = min(map_work.shape[1], cx + cw + pad)
            y1 = min(map_work.shape[0], cy + ch + pad)
            roi = map_work[y0:y1, x0:x1]
        else:
            x0, y0 = 0, 0
            roi = map_work
        if roi.size == 0:
            raise LocalizationNotConverged("Empty map ROI after coarse stage")

        # --- Fine: SIFT/ORB + RANSAC affine ---
        fine = self._fine_match(query_work, roi)
        if fine is None:
            # Fall back to coarse center if template score is strong enough.
            if coarse_score < 0.35:
                raise LocalizationNotConverged(
                    f"Fine matching failed and coarse score too low ({coarse_score:.3f})"
                )
            mx = (cx + cw / 2.0) / map_scale
            my = (cy + ch / 2.0) / map_scale
            if not np.isfinite(coarse_score):
                raise LocalizationNotConverged("Non-finite coarse score")
            confidence = float(np.clip(0.45 * coarse_score, 0.0, 0.7))
            diagnostics = {
                "method": "coarse_only",
                "coarse_score": float(coarse_score),
                "n_inliers": 0,
                "map_scale": map_scale,
            }
        else:
            M, n_inliers, inlier_ratio, residual, matches_total = fine
            affine_scale = float(np.hypot(M[0, 0], M[0, 1]))
            if affine_scale < 0.45 or affine_scale > 2.2:
                raise LocalizationNotConverged(
                    f"Implausible affine scale {affine_scale:.2f}"
                )

            # Query center in query_work coords → ROI → map_work → full map pixels.
            qcx = query_work.shape[1] / 2.0
            qcy = query_work.shape[0] / 2.0
            pt = np.array([[[qcx, qcy]]], dtype=np.float32)
            mapped = cv2.transform(pt, M)[0, 0]
            mx = (mapped[0] + x0) / map_scale
            my = (mapped[1] + y0) / map_scale

            conf_inliers = min(1.0, n_inliers / 40.0)
            conf_ratio = inlier_ratio
            conf_residual = float(np.clip(1.0 - residual / 8.0, 0.0, 1.0))
            conf_coarse = max(float(coarse_score), 0.0)
            confidence = float(
                np.clip(
                    0.30 * conf_inliers
                    + 0.30 * conf_ratio
                    + 0.20 * conf_residual
                    + 0.20 * conf_coarse,
                    0.0,
                    1.0,
                )
            )
            diagnostics = {
                "method": "coarse_plus_features",
                "coarse_score": float(coarse_score),
                "n_inliers": int(n_inliers),
                "n_matches": int(matches_total),
                "inlier_ratio": float(inlier_ratio),
                "residual_px": float(residual),
                "affine_scale": affine_scale,
                "map_scale": map_scale,
                "feature_backend": self._last_backend,
            }

            if n_inliers < self.min_inliers and coarse_score < 0.5:
                raise LocalizationNotConverged(
                    f"Too few inliers ({n_inliers}) and weak coarse score"
                )

        if confidence < self.min_confidence:
            raise LocalizationNotConverged(
                f"Confidence {confidence:.3f} below threshold {self.min_confidence}"
            )

        lat, lon = map_raster.pixel_to_latlon(mx, my)
        diagnostics.update(
            {
                "map_pixel_x": float(mx),
                "map_pixel_y": float(my),
                "yaw_applied_deg": float(-plan.yaw_deg),
                "map_source": map_raster.source,
                "map_zoom": map_raster.zoom,
                "gsd_m": plan.gsd_m,
                "rotation_matrix": rot_m.tolist() if rot_m is not None else None,
            }
        )
        return LocalizationResult(
            lat_deg=float(lat),
            lon_deg=float(lon),
            confidence=confidence,
            diagnostics=diagnostics,
        )

    _last_backend: str = "none"

    def _coarse_locate(
        self, query: np.ndarray, map_img: np.ndarray
    ) -> tuple[int, int, int, int, float] | None:
        q_gray = self._preprocess(query)
        m_gray = self._preprocess(map_img)

        # Further downscale for speed on large search areas.
        # scale = out_size / in_size  →  map_coords = small_coords / scale
        m_small, m_s = self._downscale_gray(m_gray, self.coarse_max_side)
        q_target = min(self.coarse_max_side, max(64, int(max(q_gray.shape) * m_s)))
        q_small, _ = self._downscale_gray(q_gray, q_target)

        # Query template must be smaller than the map.
        if q_small.shape[0] >= m_small.shape[0] or q_small.shape[1] >= m_small.shape[1]:
            scale = min(
                (m_small.shape[0] * 0.45) / max(q_small.shape[0], 1),
                (m_small.shape[1] * 0.45) / max(q_small.shape[1], 1),
            )
            if scale <= 0:
                return None
            q_small = cv2.resize(
                q_small,
                (
                    max(8, int(q_small.shape[1] * scale)),
                    max(8, int(q_small.shape[0] * scale)),
                ),
            )

        # Reject nearly uniform queries (no texture → unreliable match).
        if float(q_small.std()) < 2.0:
            return None

        result = cv2.matchTemplate(m_small, q_small, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if not np.isfinite(max_val):
            return None

        tw = int(round(q_small.shape[1] / m_s))
        th = int(round(q_small.shape[0] / m_s))
        mx = int(round(max_loc[0] / m_s))
        my = int(round(max_loc[1] / m_s))
        tw = max(8, min(tw, map_img.shape[1]))
        th = max(8, min(th, map_img.shape[0]))
        mx = int(np.clip(mx, 0, max(0, map_img.shape[1] - tw)))
        my = int(np.clip(my, 0, max(0, map_img.shape[0] - th)))
        return mx, my, tw, th, float(max_val)

    def _fine_match(
        self, query: np.ndarray, roi: np.ndarray
    ) -> tuple[np.ndarray, int, float, float, int] | None:
        q_gray = self._preprocess(query)
        r_gray = self._preprocess(roi)
        kp_q, des_q, backend = self._detect_describe(q_gray)
        kp_r, des_r, _ = self._detect_describe(r_gray)
        self._last_backend = backend
        if des_q is None or des_r is None or len(kp_q) < 8 or len(kp_r) < 8:
            return None

        if backend == "sift":
            matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        knn = matcher.knnMatch(des_q, des_r, k=2)
        good: list[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < self.min_inliers:
            return None

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
            return None

        inliers = int(mask.ravel().sum())
        if inliers < self.min_inliers:
            return None

        inlier_src = src[mask.ravel() == 1]
        inlier_dst = dst[mask.ravel() == 1]
        pred = cv2.transform(inlier_src, M)
        residual = float(np.linalg.norm(pred - inlier_dst, axis=2).mean())
        inlier_ratio = inliers / max(len(good), 1)
        return M, inliers, inlier_ratio, residual, len(good)

    def _detect_describe(self, gray: np.ndarray):
        sift = cv2.SIFT_create(nfeatures=4000)
        kp, des = sift.detectAndCompute(gray, None)
        if des is not None and len(kp) >= 8:
            return kp, des, "sift"
        orb = cv2.ORB_create(nfeatures=4000)
        kp, des = orb.detectAndCompute(gray, None)
        return kp, des, "orb"

    @staticmethod
    def _preprocess(bgr: np.ndarray) -> np.ndarray:
        if bgr.ndim == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _rotate_image(
        image: np.ndarray, angle_deg: float
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Rotate in-place (same canvas). Keeps optical center = image center."""
        if abs(angle_deg) < 1e-3:
            return image, None
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        return rotated, M

    @staticmethod
    def _downscale(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        side = max(h, w)
        if side <= max_side:
            return image, 1.0
        scale = max_side / side
        out = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return out, scale

    @staticmethod
    def _downscale_gray(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
        h, w = gray.shape[:2]
        side = max(h, w)
        if side <= max_side:
            return gray, 1.0
        scale = max_side / side
        out = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return out, scale
