"""Georeferencer: flight meta + image size → search plan on the map.

How to implement fully
----------------------
1. Convert altitude + intrinsics → GSD (m/px) for the query image.
2. Optionally shift the ground footprint center for non-zero pitch/roll
   (ray–plane intersection with flat-Earth or DEM).
3. Pick Web-Mercator tile zoom so map GSD ≈ query GSD.
4. Build a search bbox from (lat, lon, search_radius_m).
5. Pass yaw as a rotation prior for the localizer (optional warp).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.models import LocalizationRequest

# Equatorial meters per degree latitude (WGS84 approximation).
METERS_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class SearchPlan:
    """Geometric plan for map fetch + matching."""

    center_lat_deg: float
    center_lon_deg: float
    search_radius_m: float
    gsd_m: float
    map_zoom: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    image_width_px: int
    image_height_px: int
    hfov_deg: float
    vfov_deg: float


class Georeferencer:
    """Compute GSD, FOV, suggested tile zoom and search center."""

    def plan(self, meta: LocalizationRequest, image_size: tuple[int, int]) -> SearchPlan:
        width_px, height_px = image_size
        if width_px <= 0 or height_px <= 0:
            raise ValueError("image_size must be positive")

        hfov_deg, vfov_deg = self._fov_deg(meta, width_px, height_px)
        gsd_m = self._gsd_m(meta.altitude_m, hfov_deg, width_px)
        zoom = self._suggest_zoom(meta.lat_deg, gsd_m)

        # Skeleton: ignore pitch/roll shift; use INS lat/lon as search center.
        # Full version: intersect camera ray with ground plane.
        return SearchPlan(
            center_lat_deg=meta.lat_deg,
            center_lon_deg=meta.lon_deg,
            search_radius_m=meta.search_radius_m,
            gsd_m=gsd_m,
            map_zoom=zoom,
            yaw_deg=meta.yaw_deg,
            pitch_deg=meta.pitch_deg,
            roll_deg=meta.roll_deg,
            image_width_px=width_px,
            image_height_px=height_px,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
        )

    def _fov_deg(
        self, meta: LocalizationRequest, width_px: int, height_px: int
    ) -> tuple[float, float]:
        if meta.hfov_deg is not None:
            hfov = meta.hfov_deg
            # Assume square pixels: vfov from aspect ratio.
            vfov = math.degrees(
                2.0
                * math.atan(
                    math.tan(math.radians(hfov) / 2.0) * (height_px / width_px)
                )
            )
            return hfov, vfov

        assert meta.focal_mm and meta.sensor_w_mm and meta.sensor_h_mm
        hfov = math.degrees(
            2.0 * math.atan(meta.sensor_w_mm / (2.0 * meta.focal_mm))
        )
        vfov = math.degrees(
            2.0 * math.atan(meta.sensor_h_mm / (2.0 * meta.focal_mm))
        )
        return hfov, vfov

    @staticmethod
    def _gsd_m(altitude_m: float, hfov_deg: float, width_px: int) -> float:
        ground_width_m = 2.0 * altitude_m * math.tan(math.radians(hfov_deg) / 2.0)
        return ground_width_m / width_px

    @staticmethod
    def _suggest_zoom(lat_deg: float, gsd_m: float) -> int:
        """Pick integer zoom where Web-Mercator m/px is closest to query GSD.

        meters_per_pixel = cos(lat) * 156543.03392 / 2**zoom
        """
        cos_lat = max(math.cos(math.radians(lat_deg)), 1e-6)
        # Solve for zoom; clamp to typical imagery range.
        raw = math.log2(cos_lat * 156543.03392 / max(gsd_m, 1e-6))
        return int(min(20, max(12, round(raw))))

    @staticmethod
    def bbox_deg(plan: SearchPlan) -> tuple[float, float, float, float]:
        """Return (min_lat, min_lon, max_lat, max_lon) for the search circle."""
        d_lat = plan.search_radius_m / METERS_PER_DEG_LAT
        cos_lat = max(math.cos(math.radians(plan.center_lat_deg)), 1e-6)
        d_lon = plan.search_radius_m / (METERS_PER_DEG_LAT * cos_lat)
        return (
            plan.center_lat_deg - d_lat,
            plan.center_lon_deg - d_lon,
            plan.center_lat_deg + d_lat,
            plan.center_lon_deg + d_lon,
        )
