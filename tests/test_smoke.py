"""Smoke / unit tests (offline; no tile network required)."""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.georeferencer import METERS_PER_DEG_LAT, Georeferencer, SearchPlan
from app.localizer import Localizer
from app.main import app, pipeline
from app.map_fetcher import MapFetcher, MapRaster
from app.models import LocalizationRequest
from app.pipeline import LocalizationPipeline


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def tiny_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _pattern_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    # Add structured edges so SIFT/ORB have something to latch onto.
    for i in range(0, h, 40):
        base[i : i + 3, :] = (255, 200, 50)
    for j in range(0, w, 55):
        base[:, j : j + 3] = (50, 200, 255)
    cv2.circle(base, (w // 3, h // 3), 40, (0, 0, 255), -1)
    cv2.rectangle(base, (w // 2, h // 2), (w // 2 + 80, h // 2 + 60), (0, 255, 0), -1)
    return base


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_localize_missing_camera_400(client: TestClient, tiny_jpeg: bytes) -> None:
    r = client.post(
        "/localize",
        files={"image": ("query.jpg", tiny_jpeg, "image/jpeg")},
        data={
            "lat_deg": 55.75,
            "lon_deg": 37.62,
            "altitude_m": 300,
            "yaw_deg": 0,
            "pitch_deg": 0,
            "roll_deg": 0,
        },
    )
    assert r.status_code == 400


def test_localize_blank_image_422(client: TestClient, tiny_jpeg: bytes) -> None:
    old_fetcher = pipeline.map_fetcher
    old_localizer = pipeline.localizer
    pipeline.map_fetcher = MapFetcher(use_stub=True)
    pipeline.localizer = Localizer(min_confidence=0.25)
    try:
        r = client.post(
            "/localize",
            files={"image": ("query.jpg", tiny_jpeg, "image/jpeg")},
            data={
                "lat_deg": 55.75,
                "lon_deg": 37.62,
                "altitude_m": 300,
                "yaw_deg": 0,
                "pitch_deg": 0,
                "roll_deg": 0,
                "focal_mm": 8.0,
                "sensor_w_mm": 6.17,
                "sensor_h_mm": 4.55,
                "search_radius_m": 500,
            },
        )
        # Uniform black query vs stub pattern should fail to converge.
        assert r.status_code == 422
    finally:
        pipeline.map_fetcher = old_fetcher
        pipeline.localizer = old_localizer


def test_georeferencer_gsd_and_zoom() -> None:
    meta = LocalizationRequest(
        lat_deg=55.75,
        lon_deg=37.62,
        altitude_m=100,
        yaw_deg=0,
        pitch_deg=0,
        roll_deg=0,
        hfov_deg=60,
        search_radius_m=500,
    )
    plan = Georeferencer().plan(meta, (1920, 1080))
    assert plan.gsd_m > 0
    assert 12 <= plan.map_zoom <= 20


def test_localizer_recovers_synthetic_shift() -> None:
    """Offline: crop a patch from a synthetic map and recover its position."""
    lat0, lon0 = 55.75, 37.62
    radius_m = 400.0
    cos_lat = math.cos(math.radians(lat0))
    d_lat = radius_m / METERS_PER_DEG_LAT
    d_lon = radius_m / (METERS_PER_DEG_LAT * cos_lat)

    map_img = _pattern_image(900, 900, seed=42)
    raster = MapRaster(
        image=map_img,
        north=lat0 + d_lat,
        west=lon0 - d_lon,
        south=lat0 - d_lat,
        east=lon0 + d_lon,
        source="synthetic",
        zoom=17,
    )

    # True location slightly off center; prior further away.
    half = 120
    # Place the true patch well inside the map.
    x0, y0 = 300, 280
    query = map_img[y0 : y0 + 2 * half, x0 : x0 + 2 * half].copy()
    assert query.shape[0] == 2 * half and query.shape[1] == 2 * half
    true_lat, true_lon = raster.pixel_to_latlon(x0 + half, y0 + half)
    # Mild photometric noise.
    query = np.clip(query.astype(np.float32) + np.random.normal(0, 4, query.shape), 0, 255)
    query = query.astype(np.uint8)

    prior_lat = true_lat + 0.0015
    prior_lon = true_lon - 0.0012
    plan = SearchPlan(
        center_lat_deg=prior_lat,
        center_lon_deg=prior_lon,
        search_radius_m=radius_m,
        gsd_m=0.5,
        map_zoom=17,
        yaw_deg=0,
        pitch_deg=0,
        roll_deg=0,
        image_width_px=query.shape[1],
        image_height_px=query.shape[0],
        hfov_deg=60,
        vfov_deg=45,
    )

    result = Localizer(min_confidence=0.15, min_inliers=8).localize(query, raster, plan)
    err_m = math.hypot(
        (result.lat_deg - true_lat) * METERS_PER_DEG_LAT,
        (result.lon_deg - true_lon) * METERS_PER_DEG_LAT * cos_lat,
    )
    assert err_m < 40.0
    assert result.confidence >= 0.15


def test_pipeline_offline_stub_echo() -> None:
    pipe = LocalizationPipeline(
        map_fetcher=MapFetcher(use_stub=True),
        localizer=Localizer(echo_prior=True),
    )
    img = _pattern_image(128, 128)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    meta = LocalizationRequest(
        lat_deg=55.75,
        lon_deg=37.62,
        altitude_m=200,
        yaw_deg=0,
        pitch_deg=0,
        roll_deg=0,
        hfov_deg=60,
        search_radius_m=500,
    )
    resp = pipe.run(buf.tobytes(), meta)
    assert resp.lat_deg == pytest.approx(55.75)
    assert "gsd_m" in resp.diagnostics
