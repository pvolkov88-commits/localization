"""Orchestrates Georeferencer → MapFetcher → Localizer."""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from app.georeferencer import Georeferencer
from app.localizer import Localizer
from app.map_fetcher import MapFetcher, MapRaster
from app.models import LocalizationRequest, LocalizationResponse


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class LocalizationPipeline:
    def __init__(
        self,
        georeferencer: Georeferencer | None = None,
        map_fetcher: MapFetcher | None = None,
        localizer: Localizer | None = None,
    ) -> None:
        self.georeferencer = georeferencer or Georeferencer()
        use_stub = _env_bool("MAP_USE_STUB", False)
        self.map_fetcher = map_fetcher or MapFetcher(
            use_stub=use_stub,
            cache_dir=os.getenv("TILE_CACHE_DIR", ".cache/tiles"),
        )
        self.localizer = localizer or Localizer()
        self.last_query_bgr: np.ndarray | None = None
        self.last_map_raster: MapRaster | None = None

    def run(
        self, image_bytes: bytes, meta: LocalizationRequest
    ) -> LocalizationResponse:
        t0 = time.perf_counter()

        query_bgr = self._decode_image(image_bytes)
        h, w = query_bgr.shape[:2]

        plan = self.georeferencer.plan(meta, (w, h))
        map_raster = self.map_fetcher.fetch(plan)
        result = self.localizer.localize(query_bgr, map_raster, plan)

        self.last_query_bgr = query_bgr
        self.last_map_raster = map_raster

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        diagnostics = {
            "gsd_m": round(plan.gsd_m, 4),
            "map_zoom": map_raster.zoom,
            "planned_zoom": plan.map_zoom,
            "search_radius_m": plan.search_radius_m,
            "hfov_deg": round(plan.hfov_deg, 3),
            "vfov_deg": round(plan.vfov_deg, 3),
            "map_source": map_raster.source,
            "map_size_px": [map_raster.width, map_raster.height],
            **result.diagnostics,
        }

        return LocalizationResponse(
            crs_epsg=4326,
            lat_deg=result.lat_deg,
            lon_deg=result.lon_deg,
            delta_lat_deg=result.lat_deg - meta.lat_deg,
            delta_lon_deg=result.lon_deg - meta.lon_deg,
            confidence=result.confidence,
            processing_time_ms=round(elapsed_ms, 2),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image; expected PNG or JPEG")
        return image
