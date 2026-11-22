"""FastAPI entrypoint: GET /health, POST /localize."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app import __version__
from app.exceptions import LocalizationNotConverged, MapFetchError
from app.models import (
    ErrorResponse,
    HealthResponse,
    LocalizationRequest,
    LocalizationResponse,
)
from app.pipeline import LocalizationPipeline

app = FastAPI(
    title="UAV Visual Localization",
    version=__version__,
    description="Prototype service: refine UAV lat/lon by matching a nadir image to a map.",
)

pipeline = LocalizationPipeline()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/localize",
    response_model=LocalizationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def localize(
    image: UploadFile = File(..., description="PNG/JPEG UAV frame"),
    lat_deg: float = Form(...),
    lon_deg: float = Form(...),
    altitude_m: float = Form(...),
    yaw_deg: float = Form(...),
    pitch_deg: float = Form(...),
    roll_deg: float = Form(...),
    focal_mm: float | None = Form(None),
    sensor_w_mm: float | None = Form(None),
    sensor_h_mm: float | None = Form(None),
    hfov_deg: float | None = Form(None),
    search_radius_m: float = Form(5000.0),
) -> LocalizationResponse | JSONResponse:
    content_type = (image.content_type or "").lower()
    if content_type and content_type not in {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "application/octet-stream",
    }:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unsupported content type: {image.content_type}"},
        )

    try:
        meta = LocalizationRequest(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            altitude_m=altitude_m,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            focal_mm=focal_mm,
            sensor_w_mm=sensor_w_mm,
            sensor_h_mm=sensor_h_mm,
            hfov_deg=hfov_deg,
            search_radius_m=search_radius_m,
        )
    except Exception as exc:  # pydantic ValidationError
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    raw = await image.read()
    if not raw:
        return JSONResponse(status_code=400, content={"detail": "Empty image upload"})

    try:
        return pipeline.run(raw, meta)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except LocalizationNotConverged as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    except MapFetchError as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    except Exception as exc:  # noqa: BLE001 — last-resort 500 for prototype
        return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})
