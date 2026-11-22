"""Web Mercator tile math (XYZ / slippy map)."""

from __future__ import annotations

import math


def latlon_to_tile_xy(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat_deg)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_nw_latlon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Northwest (top-left) corner of tile → (lat, lon)."""
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def tile_bounds(
    x: int, y: int, zoom: int
) -> tuple[float, float, float, float]:
    """Return (north, west, south, east) for one tile."""
    n_lat, w_lon = tile_nw_latlon(x, y, zoom)
    s_lat, e_lon = tile_nw_latlon(x + 1, y + 1, zoom)
    return n_lat, w_lon, s_lat, e_lon


def meters_per_pixel(lat_deg: float, zoom: int) -> float:
    return math.cos(math.radians(lat_deg)) * 156543.03392 / (2**zoom)


def clamp_tile(v: int, zoom: int) -> int:
    return max(0, min(2**zoom - 1, v))
