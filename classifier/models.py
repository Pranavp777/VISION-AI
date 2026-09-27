"""
Database models for VisionClassify (PREDICTOR).

Defines `ClassificationHistory` with:
- id
- image (stored with safe UUID filenames under `media/uploads/`)
- predicted_class
- confidence (0.0 - 1.0 float)
- classification_time (seconds)
- top_predictions (JSON list of top-K predictions)
- created_at (timestamp)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from django.db import models

CLASS_EMOJI_MAP = {
    "cat": "🐱",
    "dog": "🐶",
    "fox": "🦊",
    "bird": "🐦",
    "deer": "🦌",
    "frog": "🐸",
    "horse": "🐴",
    "airplane": "✈️",
    "automobile": "🚗",
    "ship": "🚢",
    "truck": "🚚",
    "flower": "🌸",
    "person": "🧑",
    "food": "🍕",
    "electronics": "💻",
}


def safe_upload_path(instance: "ClassificationHistory", filename: str) -> str:
    """
    Generate a cryptographically safe, collision-free filename in `uploads/`
    so user-supplied filenames cannot cause path traversal or overwrite files.
    """
    ext = Path(filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        ext = ".jpg"
    unique_name = f"pred_{uuid.uuid4().hex[:16]}{ext}"
    return os.path.join("uploads", unique_name)


class ClassificationHistory(models.Model):
    """Stores each image classification request and its AI prediction metrics."""

    id = models.BigAutoField(primary_key=True)
    image = models.ImageField(
        upload_to=safe_upload_path,
        help_text="Uploaded or camera-captured image file.",
    )
    original_filename = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Sanitized display name of the uploaded file.",
    )
    predicted_class = models.CharField(
        max_length=120,
        db_index=True,
        help_text="Top predicted class label.",
    )
    confidence = models.FloatField(
        help_text="Confidence score between 0.0 and 1.0 (e.g., 0.9642).",
    )
    classification_time = models.FloatField(
        default=0.0,
        help_text="Inference processing time in seconds.",
    )
    top_predictions = models.JSONField(
        default=list,
        blank=True,
        help_text="List of top-K prediction dictionaries ({class, confidence, percentage}).",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Timestamp when the classification was performed.",
    )

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Classification History"
        verbose_name_plural = "Classification Histories"

    def __str__(self) -> str:
        return f"{self.predicted_class.title()} ({self.confidence_percentage}%) - {self.created_at:%Y-%m-%d %H:%M}"

    @property
    def confidence_percentage(self) -> float:
        """Return confidence formatted as a percentage (0.00 - 100.00)."""
        if self.confidence <= 1.0:
            return round(self.confidence * 100.0, 2)
        return round(self.confidence, 2)

    @property
    def emoji(self) -> str:
        """Return an icon/emoji matching the predicted class."""
        return CLASS_EMOJI_MAP.get(self.predicted_class.lower(), "🔍")

    def delete(self, *args, **kwargs):
        """Remove the associated image file from storage when record is deleted."""
        storage = self.image.storage if self.image else None
        image_name = self.image.name if self.image else None
        super().delete(*args, **kwargs)
        if storage and image_name and storage.exists(image_name):
            try:
                storage.delete(image_name)
            except Exception:
                pass
