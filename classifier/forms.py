"""
Forms and file validation logic for VisionClassify (PREDICTOR).

Enforces:
- Non-empty file presence
- Maximum file size limit (default 10 MB)
- Allowed file extensions (.jpg, .jpeg, .png, .webp, .bmp, .gif)
- Allowed MIME types
- Actual binary image verification using Pillow (`Image.verify()`)
"""

from __future__ import annotations

import re
from pathlib import Path

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from PIL import Image, UnidentifiedImageError


def sanitize_filename(filename: str) -> str:
    """Strip directory components and unsafe characters from user filename."""
    base = Path(filename or "capture.jpg").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return cleaned[:120] or "image.jpg"


def validate_uploaded_image(uploaded_file) -> None:
    """
    Comprehensive security and format validation for uploaded images.
    Raises ` django.core.exceptions.ValidationError` with user-friendly messages.
    """
    if not uploaded_file:
        raise ValidationError("No image selected. Please choose or capture an image first.")

    # 1. File size validation
    max_bytes = getattr(settings, "MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024)
    max_mb = getattr(settings, "MAX_UPLOAD_SIZE_MB", 10)
    if uploaded_file.size > max_bytes:
        size_mb = round(uploaded_file.size / (1024 * 1024), 2)
        raise ValidationError(
            f"Image is too large ({size_mb} MB). Maximum allowed file size is {max_mb} MB."
        )
    if uploaded_file.size == 0:
        raise ValidationError("The selected image file is empty. Please choose a valid image.")

    # 2. Extension validation
    ext = Path(uploaded_file.name or "").suffix.lower()
    allowed_exts = getattr(
        settings,
        "ALLOWED_IMAGE_EXTENSIONS",
        {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"},
    )
    if ext not in allowed_exts:
        allowed_list = ", ".join(sorted(e.lstrip(".").upper() for e in allowed_exts))
        raise ValidationError(
            f"Unsupported image format '{ext or 'unknown'}'. Supported formats: {allowed_list}."
        )

    # 3. Content-Type / MIME validation
    content_type = getattr(uploaded_file, "content_type", "") or ""
    allowed_mimes = getattr(
        settings,
        "ALLOWED_IMAGE_MIME_TYPES",
        {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/gif"},
    )
    if content_type and content_type.lower() not in allowed_mimes:
        raise ValidationError(
            f"Unsupported file MIME type '{content_type}'. Please upload a valid image file."
        )

    # 4. Deep binary header verification via Pillow
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            img.verify()
        uploaded_file.seek(0)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        uploaded_file.seek(0)
        raise ValidationError(
            "Invalid or corrupted image file. Please select a valid photo (JPG, PNG, WEBP, BMP, or GIF)."
        ) from exc


class ImageUploadForm(forms.Form):
    """Form for uploading or capturing an image for AI classification."""

    image = forms.ImageField(
        required=True,
        validators=[validate_uploaded_image],
        error_messages={
            "required": "No image selected. Please upload an image or use your camera.",
            "invalid": "Invalid image file. Please choose a valid photo.",
            "invalid_image": "The uploaded file is not a valid image or is corrupted.",
            "empty": "The uploaded file is empty.",
        },
        widget=forms.ClearableFileInput(
            attrs={
                "id": "imageInput",
                "accept": "image/*",
                "class": "sr-only-input",
                "aria-label": "Upload image from device",
            }
        ),
    )

    def clean_image(self):
        image = self.cleaned_data.get("image")
        validate_uploaded_image(image)
        return image
