"""Pydantic schemas for /localize request metadata and JSON response."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class LocalizationRequest(BaseModel):
    """Flight / camera metadata from multipart form fields (without the image)."""

    lat_deg: float = Field(..., description="Initial latitude WGS84")
    lon_deg: float = Field(..., description="Initial longitude WGS84")
    altitude_m: float = Field(..., ge=100.0, le=800.0, description="Flight altitude, m")
    yaw_deg: float = Field(..., description="Yaw / heading, degrees")
    pitch_deg: float = Field(..., description="Pitch, degrees (0 ≈ nadir)")
    roll_deg: float = Field(..., description="Roll, degrees")

    focal_mm: float | None = Field(None, gt=0, description="Focal length, mm")
    sensor_w_mm: float | None = Field(None, gt=0, description="Sensor width, mm")
    sensor_h_mm: float | None = Field(None, gt=0, description="Sensor height, mm")
    hfov_deg: float | None = Field(None, gt=0, lt=180, description="Horizontal FOV alternative")

    search_radius_m: float = Field(5000.0, gt=0, le=20_000, description="Search radius, m")

    @model_validator(mode="after")
    def require_camera_intrinsics(self) -> LocalizationRequest:
        has_focal_sensor = (
            self.focal_mm is not None
            and self.sensor_w_mm is not None
            and self.sensor_h_mm is not None
        )
        has_hfov = self.hfov_deg is not None
        if not has_focal_sensor and not has_hfov:
            raise ValueError(
                "Provide either (focal_mm, sensor_w_mm, sensor_h_mm) or hfov_deg"
            )
        return self


class Diagnostics(BaseModel):
    """Optional debug fields; free-form for the prototype."""

    model_config = {"extra": "allow"}

    gsd_m: float | None = None
    map_zoom: int | None = None
    search_radius_m: float | None = None
    map_source: str | None = None
    n_inliers: int | None = None
    note: str | None = None


class LocalizationResponse(BaseModel):
    crs_epsg: int = 4326
    lat_deg: float
    lon_deg: float
    delta_lat_deg: float
    delta_lon_deg: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"


class ErrorResponse(BaseModel):
    detail: str
