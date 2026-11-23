"""Map Fetcher: download & stitch Esri World Imagery (or stub) tiles."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np

from app.exceptions import MapFetchError
from app.georeferencer import METERS_PER_DEG_LAT, SearchPlan
from app.tiles import clamp_tile, latlon_to_tile_xy, tile_nw_latlon

logger = logging.getLogger(__name__)

# Esri World Imagery — publicly tileable; attribution required in README.
ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

TILE_SIZE = 256
DEFAULT_MAX_TILES_PER_SIDE = 16  # 16×16 × 256 ≈ 4096 px
USER_AGENT = "uav-visual-localization-prototype/0.1 (educational)"


@dataclass
class MapRaster:
    """Georeferenced map patch (axis-aligned Web-Mercator stitch ≈ EPSG:4326 corners)."""

    image: np.ndarray  # HxWx3 uint8 BGR (OpenCV convention)
    north: float
    west: float
    south: float
    east: float
    source: str
    zoom: int

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def pixel_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        # Lon is linear in Web Mercator X; lat must be interpolated in mercator-Y.
        lon = self.west + (x / max(self.width, 1)) * (self.east - self.west)
        merc_n = _lat_to_mercator_y(self.north)
        merc_s = _lat_to_mercator_y(self.south)
        merc = merc_n + (y / max(self.height, 1)) * (merc_s - merc_n)
        return _mercator_y_to_lat(merc), lon

    def latlon_to_pixel(self, lat: float, lon: float) -> tuple[float, float]:
        x = (lon - self.west) / max(self.east - self.west, 1e-15) * self.width
        merc_n = _lat_to_mercator_y(self.north)
        merc_s = _lat_to_mercator_y(self.south)
        merc = _lat_to_mercator_y(lat)
        # Keep the sign of (merc_s - merc_n): north > south ⇒ negative delta.
        # max(..., 1e-15) would clamp that to +1e-15 and explode y.
        den = merc_s - merc_n
        if abs(den) < 1e-15:
            den = 1e-15
        y = (merc - merc_n) / den * self.height
        return x, y


def _lat_to_mercator_y(lat_deg: float) -> float:
    return math.asinh(math.tan(math.radians(lat_deg)))


def _mercator_y_to_lat(merc_y: float) -> float:
    return math.degrees(math.atan(math.sinh(merc_y)))


class MapFetcher:
    """Fetch map imagery for a SearchPlan with on-disk tile cache."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/tiles",
        use_stub: bool = False,
        source_name: str = "esri-world-imagery",
        tile_url: str = ESRI_WORLD_IMAGERY,
        max_tiles_per_side: int = DEFAULT_MAX_TILES_PER_SIDE,
        timeout_s: float = 30.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_stub = use_stub
        self.source_name = source_name
        self.tile_url = tile_url
        self.max_tiles_per_side = max_tiles_per_side
        self.timeout_s = timeout_s

    def fetch(self, plan: SearchPlan) -> MapRaster:
        if self.use_stub:
            return self._stub_raster(plan)
        try:
            return self._fetch_tiles(plan)
        except MapFetchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MapFetchError(f"Failed to fetch map tiles: {exc}") from exc

    def fetch_bbox(
        self,
        north: float,
        west: float,
        south: float,
        east: float,
        zoom: int,
    ) -> MapRaster:
        """Fetch an explicit lat/lon bbox (used by example generator)."""
        if self.use_stub:
            h = w = 512
            return MapRaster(
                image=np.zeros((h, w, 3), dtype=np.uint8),
                north=north,
                west=west,
                south=south,
                east=east,
                source=self.source_name,
                zoom=zoom,
            )
        lat_mid = 0.5 * (north + south)
        zoom = self._fit_zoom(lat_mid, west, east, south, north, zoom)
        return self._stitch_bbox(north, west, south, east, zoom)

    def _fetch_tiles(self, plan: SearchPlan) -> MapRaster:
        cos_lat = max(math.cos(math.radians(plan.center_lat_deg)), 1e-6)
        d_lat = plan.search_radius_m / METERS_PER_DEG_LAT
        d_lon = plan.search_radius_m / (METERS_PER_DEG_LAT * cos_lat)
        north = plan.center_lat_deg + d_lat
        south = plan.center_lat_deg - d_lat
        west = plan.center_lon_deg - d_lon
        east = plan.center_lon_deg + d_lon

        zoom = self._fit_zoom(plan.center_lat_deg, west, east, south, north, plan.map_zoom)
        return self._stitch_bbox(north, west, south, east, zoom)

    def _fit_zoom(
        self,
        lat: float,
        west: float,
        east: float,
        south: float,
        north: float,
        preferred: int,
    ) -> int:
        zoom = preferred
        while zoom > 10:
            x0, y0 = latlon_to_tile_xy(north, west, zoom)
            x1, y1 = latlon_to_tile_xy(south, east, zoom)
            n_x = abs(x1 - x0) + 1
            n_y = abs(y1 - y0) + 1
            if n_x <= self.max_tiles_per_side and n_y <= self.max_tiles_per_side:
                return zoom
            zoom -= 1
        return zoom

    def _stitch_bbox(
        self,
        north: float,
        west: float,
        south: float,
        east: float,
        zoom: int,
    ) -> MapRaster:
        x0, y0 = latlon_to_tile_xy(north, west, zoom)
        x1, y1 = latlon_to_tile_xy(south, east, zoom)
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)

        n_x = x_max - x_min + 1
        n_y = y_max - y_min + 1
        if n_x * n_y > self.max_tiles_per_side**2:
            raise MapFetchError(
                f"Tile grid too large ({n_x}x{n_y}) at zoom={zoom}; reduce search_radius"
            )

        canvas = np.zeros((n_y * TILE_SIZE, n_x * TILE_SIZE, 3), dtype=np.uint8)
        with httpx.Client(
            timeout=self.timeout_s,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as client:
            for ty in range(y_min, y_max + 1):
                for tx in range(x_min, x_max + 1):
                    tile = self._load_tile(client, zoom, tx, ty)
                    row = ty - y_min
                    col = tx - x_min
                    canvas[
                        row * TILE_SIZE : (row + 1) * TILE_SIZE,
                        col * TILE_SIZE : (col + 1) * TILE_SIZE,
                    ] = tile

        # Exact geographic bounds of the stitched tile grid (NW of first, SE of last+1).
        nw_lat, nw_lon = tile_nw_latlon(x_min, y_min, zoom)
        se_lat, se_lon = tile_nw_latlon(x_max + 1, y_max + 1, zoom)

        return MapRaster(
            image=canvas,
            north=nw_lat,
            west=nw_lon,
            south=se_lat,
            east=se_lon,
            source=self.source_name,
            zoom=zoom,
        )

    def _load_tile(self, client: httpx.Client, z: int, x: int, y: int) -> np.ndarray:
        x = clamp_tile(x, z)
        y = clamp_tile(y, z)
        cache_path = self.cache_dir / self.source_name / str(z) / str(x) / f"{y}.jpg"
        if cache_path.exists():
            data = np.fromfile(cache_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None and img.shape[0] == TILE_SIZE:
                return img

        url = self.tile_url.format(z=z, x=x, y=y)
        try:
            resp = client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MapFetchError(f"Tile download failed {url}: {exc}") from exc

        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise MapFetchError(f"Cannot decode tile {z}/{x}/{y}")
        if img.shape[0] != TILE_SIZE or img.shape[1] != TILE_SIZE:
            img = cv2.resize(img, (TILE_SIZE, TILE_SIZE))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        return img

    def _stub_raster(self, plan: SearchPlan) -> MapRaster:
        cos_lat = max(math.cos(math.radians(plan.center_lat_deg)), 1e-6)
        d_lat = plan.search_radius_m / METERS_PER_DEG_LAT
        d_lon = plan.search_radius_m / (METERS_PER_DEG_LAT * cos_lat)
        side_m = 2.0 * plan.search_radius_m
        size_px = int(max(64, min(2048, side_m / max(plan.gsd_m, 0.05))))
        # Structured pattern so matching tests can work offline.
        yy, xx = np.mgrid[0:size_px, 0:size_px]
        image = np.stack(
            [
                ((xx * 3 + yy) % 256).astype(np.uint8),
                ((xx * 7) % 256).astype(np.uint8),
                ((yy * 5) % 256).astype(np.uint8),
            ],
            axis=-1,
        )
        return MapRaster(
            image=image,
            north=plan.center_lat_deg + d_lat,
            west=plan.center_lon_deg - d_lon,
            south=plan.center_lat_deg - d_lat,
            east=plan.center_lon_deg + d_lon,
            source="stub-pattern",
            zoom=plan.map_zoom,
        )
