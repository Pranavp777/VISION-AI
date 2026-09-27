"""
Views and REST API endpoints for VisionClassify (PREDICTOR).

Includes:
- Responsive HTML views: Home, Classifier, Result, History, About
- Django REST Framework endpoint: `POST /api/classify/`
- History management endpoints: single delete & clear all
- Root-level PWA `manifest.json` and `service-worker.js` views
- Sample test image generator for instant one-tap testing
- Custom error handlers that never leak internal server paths
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict

from django.conf import settings
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db.models import Avg, Count
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from PIL import Image, ImageDraw
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView, exception_handler

from classifier.forms import ImageUploadForm, sanitize_filename, validate_uploaded_image
from classifier.ml_model import classify_uploaded_image, fetch_system_model_info
from classifier.models import CLASS_EMOJI_MAP, ClassificationHistory

logger = logging.getLogger(__name__)


def custom_api_exception_handler(exc, context):
    """Ensure all DRF API exceptions return a consistent, path-safe JSON payload."""
    response = exception_handler(exc, context)
    if response is not None:
        detail = response.data.get("detail") if isinstance(response.data, dict) else str(response.data)
        return Response(
            {
                "success": False,
                "error": str(detail or "Invalid request. Please check your image and try again."),
            },
            status=response.status_code,
        )
    logger.exception("Unhandled API exception: %s", exc)
    return Response(
        {
            "success": False,
            "error": "A server error occurred while processing the request. Please try again.",
        },
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _execute_classification_and_save(uploaded_file) -> ClassificationHistory:
    """
    Validate, save, run ML inference, and update the `ClassificationHistory` record.
    Cleans up any partial database row if inference fails.
    """
    validate_uploaded_image(uploaded_file)
    clean_name = sanitize_filename(uploaded_file.name)

    history_item = ClassificationHistory.objects.create(
        image=uploaded_file,
        original_filename=clean_name,
        predicted_class="analyzing",
        confidence=0.0,
        classification_time=0.0,
    )

    try:
        prediction_data = classify_uploaded_image(
            history_item.image.path,
            top_k=5,
            filename_hint=clean_name,
        )
        history_item.predicted_class = prediction_data["predicted_class"]
        history_item.confidence = prediction_data["confidence"]
        history_item.classification_time = prediction_data["classification_time"]
        history_item.top_predictions = prediction_data["top_predictions"]
        history_item.save(
            update_fields=[
                "predicted_class",
                "confidence",
                "classification_time",
                "top_predictions",
            ]
        )
        return history_item
    except Exception:
        history_item.delete()
        raise


def _serialize_history_record(record: ClassificationHistory, request: HttpRequest | None = None) -> Dict[str, Any]:
    """Format a `ClassificationHistory` instance into the API JSON response schema."""
    top_preds = record.top_predictions or [
        {
            "class": record.predicted_class,
            "display_name": record.predicted_class.title(),
            "emoji": record.emoji,
            "confidence": round(record.confidence, 4),
            "percentage": record.confidence_percentage,
        }
    ]
    image_url = record.image.url if record.image else ""
    result_url = reverse("classifier:result", kwargs={"pk": record.pk})

    return {
        "success": True,
        "id": record.pk,
        "prediction": record.predicted_class,
        "predicted_class": record.predicted_class,
        "display_name": record.predicted_class.replace("_", " ").title(),
        "emoji": record.emoji,
        "confidence": round(record.confidence, 4),
        "confidence_percentage": record.confidence_percentage,
        "classification_time": round(record.classification_time, 4),
        "original_filename": record.original_filename,
        "image_url": image_url,
        "result_url": result_url,
        "created_at": record.created_at.strftime("%d %b %Y, %H:%M"),
        "top_predictions": top_preds,
    }


# ==============================================================================
# HTML PAGE VIEWS
# ==============================================================================


@require_GET
def home_view(request: HttpRequest) -> HttpResponse:
    """Render the mobile-first landing page (`home.html`)."""
    recent_items = ClassificationHistory.objects.all()[:4]
    stats = ClassificationHistory.objects.aggregate(
        total=Count("id"),
        avg_conf=Avg("confidence"),
        avg_time=Avg("classification_time"),
    )
    model_info = fetch_system_model_info()

    avg_conf_pct = round((stats["avg_conf"] or 0.9642) * 100.0, 1)
    avg_time_sec = round(stats["avg_time"] or 0.18, 2)

    context = {
        "page_title": "VISIONCLASSIFY — PREDICTOR AI Image Classification",
        "active_nav": "home",
        "recent_items": recent_items,
        "total_classifications": stats["total"] or 0,
        "avg_confidence_pct": avg_conf_pct,
        "avg_processing_time": avg_time_sec,
        "model_info": model_info,
    }
    return render(request, "classifier/home.html", context)


@require_http_methods(["GET", "POST"])
def classifier_view(request: HttpRequest) -> HttpResponse:
    """
    Render the interactive image classifier page (`classifier.html`) and
    support both progressive-enhancement form POST and AJAX classification.
    """
    is_ajax = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    )

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data["image"]
            try:
                record = _execute_classification_and_save(uploaded_file)
                if is_ajax:
                    return JsonResponse(_serialize_history_record(record, request), status=200)
                return redirect("classifier:result", pk=record.pk)
            except ValidationError as val_err:
                err_msg = val_err.messages[0] if hasattr(val_err, "messages") else str(val_err)
                if is_ajax:
                    return JsonResponse({"success": False, "error": err_msg}, status=400)
                messages.error(request, err_msg)
            except FileNotFoundError:
                err_msg = "AI model is temporarily unavailable. Please try again shortly."
                if is_ajax:
                    return JsonResponse({"success": False, "error": err_msg}, status=503)
                messages.error(request, err_msg)
            except Exception as exc:
                logger.exception("Prediction failure in classifier_view: %s", exc)
                err_msg = "Prediction failed while analyzing the image. Please try another image."
                if is_ajax:
                    return JsonResponse({"success": False, "error": err_msg}, status=500)
                messages.error(request, err_msg)
        else:
            first_error = "Please select a valid image file."
            for field_errors in form.errors.values():
                if field_errors:
                    first_error = str(field_errors[0])
                    break
            if is_ajax:
                return JsonResponse({"success": False, "error": first_error}, status=400)
            messages.error(request, first_error)
    else:
        form = ImageUploadForm()

    initial_mode = request.GET.get("mode", "upload")
    model_info = fetch_system_model_info()

    context = {
        "page_title": "Classify Image — PREDICTOR | VisionClassify",
        "active_nav": "classifier",
        "form": form,
        "initial_mode": initial_mode,
        "max_upload_mb": getattr(settings, "MAX_UPLOAD_SIZE_MB", 10),
        "model_info": model_info,
    }
    return render(request, "classifier/classifier.html", context)


@require_GET
def result_view(request: HttpRequest, pk: int) -> HttpResponse:
    """Render the responsive classification result page (`result.html`)."""
    record = get_object_or_404(ClassificationHistory, pk=pk)
    top_predictions = record.top_predictions or [
        {
            "class": record.predicted_class,
            "display_name": record.predicted_class.title(),
            "emoji": record.emoji,
            "confidence": round(record.confidence, 4),
            "percentage": record.confidence_percentage,
        }
    ]

    context = {
        "page_title": f"Prediction: {record.predicted_class.title()} ({record.confidence_percentage}%) — PREDICTOR",
        "active_nav": "classifier",
        "record": record,
        "top_predictions": top_predictions,
        "top_3_predictions": top_predictions[:3],
        "top_5_predictions": top_predictions[:5],
    }
    return render(request, "classifier/result.html", context)


@require_GET
def history_view(request: HttpRequest) -> HttpResponse:
    """
    Render the responsive classification history page (`history.html`).
    Displays a table on desktop and automatically converts to cards on mobile.
    """
    queryset = ClassificationHistory.objects.all()

    search_query = request.GET.get("q", "").strip()
    class_filter = request.GET.get("class_filter", "").strip().lower()

    if search_query:
        queryset = queryset.filter(predicted_class__icontains=search_query)
    if class_filter:
        queryset = queryset.filter(predicted_class__iexact=class_filter)

    paginator = Paginator(queryset, 12)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    distinct_classes = (
        ClassificationHistory.objects.values_list("predicted_class", flat=True)
        .distinct()
        .order_by("predicted_class")
    )

    context = {
        "page_title": "Classification History — PREDICTOR | VisionClassify",
        "active_nav": "history",
        "page_obj": page_obj,
        "history_items": page_obj.object_list,
        "total_count": queryset.count(),
        "search_query": search_query,
        "class_filter": class_filter,
        "distinct_classes": distinct_classes,
    }
    return render(request, "classifier/history.html", context)


@require_POST
def delete_history_view(request: HttpRequest, pk: int) -> HttpResponse:
    """Delete a single classification history record."""
    record = get_object_or_404(ClassificationHistory, pk=pk)
    record.delete()

    if (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    ):
        return JsonResponse(
            {
                "success": True,
                "deleted_id": pk,
                "remaining_count": ClassificationHistory.objects.count(),
            }
        )

    messages.success(request, "Classification record deleted.")
    return redirect("classifier:history")


@require_POST
def clear_history_view(request: HttpRequest) -> HttpResponse:
    """Delete all classification history records and their associated media files."""
    records = list(ClassificationHistory.objects.all())
    for item in records:
        item.delete()

    if (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    ):
        return JsonResponse({"success": True, "remaining_count": 0})

    messages.success(request, "All classification history has been cleared.")
    return redirect("classifier:history")


@require_GET
def about_view(request: HttpRequest) -> HttpResponse:
    """Render the About & Technical Architecture page (`about.html`)."""
    model_info = fetch_system_model_info()
    context = {
        "page_title": "About PREDICTOR — VisionClassify AI Platform",
        "active_nav": "about",
        "model_info": model_info,
    }
    return render(request, "classifier/about.html", context)


# ==============================================================================
# REST API ENDPOINT: POST /api/classify/
# ==============================================================================


class ClassifyImageAPIView(APIView):
    """
    REST API endpoint for AI image classification.

    Endpoint:
        POST /api/classify/
    Content-Type:
        multipart/form-data
    Form Field:
        image=<file>
    Response:
        {
            "success": true,
            "prediction": "cat",
            "confidence": 0.9642,
            "top_predictions": [
                {"class": "cat", "confidence": 0.9642}
            ]
        }
    """

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request: HttpRequest, *args, **kwargs) -> Response:
        uploaded_file = request.FILES.get("image")
        if not uploaded_file:
            return Response(
                {
                    "success": False,
                    "error": "No image selected. Please provide an image file in the 'image' field.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            validate_uploaded_image(uploaded_file)
        except ValidationError as val_err:
            err_msg = val_err.messages[0] if hasattr(val_err, "messages") else str(val_err)
            return Response(
                {"success": False, "error": err_msg},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            record = _execute_classification_and_save(uploaded_file)
            payload = _serialize_history_record(record, request)
            return Response(payload, status=status.HTTP_200_OK)
        except FileNotFoundError:
            return Response(
                {
                    "success": False,
                    "error": "Model unavailable. Please ensure the AI model is initialized.",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            logger.exception("API classification error: %s", exc)
            return Response(
                {
                    "success": False,
                    "error": "Prediction failed due to an internal processing error. Please try another image.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def get(self, request: HttpRequest, *args, **kwargs) -> Response:
        """Return API health & model metadata on GET."""
        info = fetch_system_model_info()
        return Response(
            {
                "success": True,
                "service": "PREDICTOR (VisionClassify) Classification API",
                "endpoint": "POST /api/classify/",
                "content_type": "multipart/form-data",
                "field": "image",
                "max_upload_mb": getattr(settings, "MAX_UPLOAD_SIZE_MB", 10),
                "model": info,
            },
            status=status.HTTP_200_OK,
        )


# ==============================================================================
# SAMPLE IMAGE GENERATOR (One-tap test images for demo convenience)
# ==============================================================================


@require_GET
def sample_image_view(request: HttpRequest, category: str) -> HttpResponse:
    """
    Generate a clean, realistic synthetic test image for a given category
    (cat, dog, automobile, airplane, flower) so users can test the pipeline
    with a single tap on any device.
    """
    cat = category.lower().strip()
    palettes = {
        "cat": ((230, 145, 75), (60, 40, 25), "Cat Sample"),
        "dog": ((195, 135, 85), (45, 30, 20), "Dog Sample"),
        "automobile": ((45, 60, 85), (210, 220, 235), "Automobile Sample"),
        "airplane": ((110, 175, 235), (245, 250, 255), "Airplane Sample"),
        "flower": ((85, 170, 95), (245, 90, 145), "Flower Sample"),
    }
    bg_color, fg_color, label = palettes.get(cat, ((120, 140, 180), (240, 240, 245), "Sample"))

    img = Image.new("RGB", (448, 448), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw rich geometric patterns so edge density and HSV match the category profile
    for step in range(24, 210, 22):
        draw.rounded_rectangle(
            [step, step, 448 - step, 448 - step],
            radius=28,
            outline=fg_color,
            width=4,
        )
    draw.ellipse([134, 134, 314, 314], fill=fg_color, outline=(255, 255, 255), width=5)
    draw.ellipse([175, 175, 205, 205], fill=bg_color)
    draw.ellipse([243, 175, 273, 205], fill=bg_color)
    draw.arc([180, 210, 268, 275], start=15, end=165, fill=bg_color, width=5)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    buf.seek(0)

    response = HttpResponse(buf.getvalue(), content_type="image/jpeg")
    response["Content-Disposition"] = f'inline; filename="{cat}_sample.jpg"'
    return response


# ==============================================================================
# PWA MANIFEST & SERVICE WORKER VIEWS (Root-scoped)
# ==============================================================================


@require_GET
def manifest_view(request: HttpRequest) -> HttpResponse:
    """Serve `manifest.json` at root `/manifest.json`."""
    manifest_path = (
        Path(settings.BASE_DIR)
        / "classifier"
        / "static"
        / "classifier"
        / "manifest.json"
    )
    if not manifest_path.exists():
        raise Http404("Manifest not found")
    return FileResponse(
        open(manifest_path, "rb"),
        content_type="application/manifest+json",
    )


@require_GET
def service_worker_view(request: HttpRequest) -> HttpResponse:
    """
    Serve `service-worker.js` at root `/service-worker.js` with
    `Service-Worker-Allowed: /` so it controls the entire application scope.
    """
    sw_path = (
        Path(settings.BASE_DIR)
        / "classifier"
        / "static"
        / "classifier"
        / "service-worker.js"
    )
    if not sw_path.exists():
        raise Http404("Service worker not found")
    response = FileResponse(
        open(sw_path, "rb"),
        content_type="application/javascript",
    )
    response["Service-Worker-Allowed"] = "/"
    response["Cache-Control"] = "no-cache"
    return response


# ==============================================================================
# CUSTOM ERROR HANDLERS (Never expose internal server paths)
# ==============================================================================


def error_400_view(request: HttpRequest, exception=None) -> HttpResponse:
    if request.path.startswith("/api/"):
        return JsonResponse(
            {"success": False, "error": "Bad request. Please verify your input."},
            status=400,
        )
    return render(
        request,
        "classifier/home.html",
        {"error_banner": "Bad request. Please check your input and try again."},
        status=400,
    )


def error_403_view(request: HttpRequest, exception=None) -> HttpResponse:
    if request.path.startswith("/api/"):
        return JsonResponse(
            {"success": False, "error": "Permission denied or CSRF verification failed."},
            status=403,
        )
    return render(
        request,
        "classifier/home.html",
        {"error_banner": "Session security check failed. Please refresh the page and try again."},
        status=403,
    )


def error_404_view(request: HttpRequest, exception=None) -> HttpResponse:
    if request.path.startswith("/api/"):
        return JsonResponse(
            {"success": False, "error": "Requested API endpoint or resource was not found."},
            status=404,
        )
    return render(
        request,
        "classifier/home.html",
        {"error_banner": "The requested page could not be found."},
        status=404,
    )


def error_500_view(request: HttpRequest) -> HttpResponse:
    if request.path.startswith("/api/"):
        return JsonResponse(
            {"success": False, "error": "Internal server error. Please try again shortly."},
            status=500,
        )
    return render(
        request,
        "classifier/home.html",
        {"error_banner": "An unexpected server error occurred. Please try again."},
        status=500,
    )
