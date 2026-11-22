"""Build side-by-side visualization: query | map with prior/result markers."""

from __future__ import annotations

import cv2
import numpy as np

from app.map_fetcher import MapRaster


def make_visualization(
    query_bgr: np.ndarray,
    map_raster: MapRaster,
    prior_lat: float,
    prior_lon: float,
    result_lat: float | None,
    result_lon: float | None,
    panel_h: int = 480,
) -> np.ndarray:
    q = _resize_to_height(query_bgr, panel_h)
    m = _resize_to_height(map_raster.image, panel_h)

    overlay = m.copy()
    scale_x = float(m.shape[1]) / float(map_raster.width)
    scale_y = float(m.shape[0]) / float(map_raster.height)

    px, py = map_raster.latlon_to_pixel(prior_lat, prior_lon)
    p_prior = _clamp_point(px * scale_x, py * scale_y, overlay.shape[1], overlay.shape[0])
    _draw_cross(overlay, p_prior, (0, 165, 255), "prior")

    if result_lat is not None and result_lon is not None:
        rx, ry = map_raster.latlon_to_pixel(result_lat, result_lon)
        p_result = _clamp_point(
            rx * scale_x, ry * scale_y, overlay.shape[1], overlay.shape[0]
        )
        cv2.arrowedLine(overlay, p_prior, p_result, (0, 255, 0), 2, tipLength=0.15)
        cv2.circle(overlay, p_result, 8, (0, 255, 0), 2)
        cv2.putText(
            overlay,
            "result",
            (p_result[0] + 8, max(16, p_result[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    h = panel_h
    q_panel = _pad_to(q, h, max(q.shape[1], 1))
    m_panel = _pad_to(overlay, h, max(overlay.shape[1], 1))
    gap = np.full((h, 8, 3), 30, dtype=np.uint8)
    canvas = np.hstack([q_panel, gap, m_panel])

    cv2.putText(
        canvas, "query", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "map",
        (q_panel.shape[1] + 20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_cross(img: np.ndarray, pt: tuple[int, int], color: tuple[int, int, int], label: str) -> None:
    x, y = pt
    cv2.line(img, (x - 12, y), (x + 12, y), color, 2)
    cv2.line(img, (x, y - 12), (x, y + 12), color, 2)
    cv2.putText(
        img,
        label,
        (x + 8, max(16, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def _clamp_point(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    xi = int(float(np.round(x)))
    yi = int(float(np.round(y)))
    xi = max(0, min(w - 1, xi))
    yi = max(0, min(h - 1, yi))
    return (xi, yi)


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = height / max(h, 1)
    return cv2.resize(image, (max(1, int(w * scale)), height), interpolation=cv2.INTER_AREA)


def _pad_to(image: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    hh, ww = image.shape[:2]
    out[: min(hh, h), : min(ww, w)] = image[: min(hh, h), : min(ww, w)]
    return out
