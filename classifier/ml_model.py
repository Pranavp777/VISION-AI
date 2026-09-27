"""
Django-to-ML Bridge (`classifier/ml_model.py`).

Provides high-level helper functions for the `classifier` Django views to:
- Run image classification via `ml.predictor.predict_image`
- Enrich predictions with class emojis and formatted percentages
- Compress overly large high-resolution camera captures safely before inference
- Query model readiness status
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageOps

from classifier.models import CLASS_EMOJI_MAP
from ml.predictor import get_model_status, predict_image

logger = logging.getLogger(__name__)

MAX_DIMENSION_PX = 1600


def optimize_stored_image(image_path: str | Path) -> None:
    """
    Optimize high-resolution mobile camera images on disk so they load quickly
    on mobile networks while preserving visual clarity and aspect ratio.
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        return

    try:
        with Image.open(path_obj) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            if max(width, height) > MAX_DIMENSION_PX:
                img.thumbnail((MAX_DIMENSION_PX, MAX_DIMENSION_PX), Image.Resampling.LANCZOS)
                ext = path_obj.suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    img.convert("RGB").save(path_obj, format="JPEG", quality=86, optimize=True)
                elif ext == ".webp":
                    img.save(path_obj, format="WEBP", quality=86, method=4)
                elif ext == ".png":
                    img.save(path_obj, format="PNG", optimize=True)
    except Exception as exc:
        logger.warning("Non-fatal image optimization warning for %s: %s", path_obj.name, exc)


def classify_uploaded_image(
    image_path: str | Path,
    top_k: int = 5,
    filename_hint: str = "",
) -> Dict[str, Any]:
    """
    Optimize and classify an uploaded image, returning enriched prediction data
    ready for both HTML templates and the REST API.
    """
    optimize_stored_image(image_path)
    result = predict_image(image_path, top_k=top_k, filename_hint=filename_hint)

    predicted_class = str(result.get("predicted_class", "unknown"))
    confidence = float(result.get("confidence", 0.0))
    confidence_pct = round(confidence * 100.0, 2) if confidence <= 1.0 else round(confidence, 2)

    enriched_top = []
    for item in result.get("top_predictions", []):
        cls_label = str(item.get("class", "unknown"))
        conf_val = float(item.get("confidence", 0.0))
        pct_val = round(conf_val * 100.0, 2) if conf_val <= 1.0 else round(conf_val, 2)
        enriched_top.append(
            {
                "class": cls_label,
                "display_name": cls_label.replace("_", " ").title(),
                "emoji": CLASS_EMOJI_MAP.get(cls_label.lower(), "🔍"),
                "confidence": round(conf_val, 4),
                "percentage": pct_val,
            }
        )

    return {
        "predicted_class": predicted_class,
        "display_name": predicted_class.replace("_", " ").title(),
        "emoji": CLASS_EMOJI_MAP.get(predicted_class.lower(), "🔍"),
        "confidence": round(confidence, 4),
        "confidence_percentage": confidence_pct,
        "top_predictions": enriched_top,
        "classification_time": float(result.get("classification_time", 0.0)),
        "image_metadata": result.get("image_metadata", {}),
    }


def fetch_system_model_info() -> Dict[str, Any]:
    """Return model status enriched with emoji mappings for the UI."""
    info = get_model_status()
    classes_with_icons = [
        {
            "name": c,
            "display_name": c.replace("_", " ").title(),
            "emoji": CLASS_EMOJI_MAP.get(c.lower(), "🔍"),
        }
        for c in info.get("classes", [])
    ]
    info["classes_enriched"] = classes_with_icons
    return info
