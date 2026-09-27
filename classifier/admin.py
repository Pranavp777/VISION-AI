"""
Django Admin configuration for VisionClassify (PREDICTOR).
"""

from django.contrib import admin
from django.utils.html import format_html

from classifier.models import ClassificationHistory


@admin.register(ClassificationHistory)
class ClassificationHistoryAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "thumbnail_preview",
        "predicted_class_badge",
        "confidence_display",
        "classification_time_display",
        "created_at",
    )
    list_filter = ("predicted_class", "created_at")
    search_fields = ("predicted_class", "original_filename")
    readonly_fields = (
        "id",
        "thumbnail_preview",
        "created_at",
        "classification_time",
        "top_predictions",
    )
    ordering = ("-created_at",)
    list_per_page = 25

    def thumbnail_preview(self, obj: ClassificationHistory) -> str:
        if obj.image:
            return format_html(
                '<img src="{}" alt="{}" style="width:52px;height:52px;object-fit:cover;border-radius:8px;border:1px solid #cbd5e1;" />',
                obj.image.url,
                obj.predicted_class,
            )
        return "No Image"

    thumbnail_preview.short_description = "Preview"

    def predicted_class_badge(self, obj: ClassificationHistory) -> str:
        return format_html(
            "<strong>{} {}</strong>",
            obj.emoji,
            obj.predicted_class.title(),
        )

    predicted_class_badge.short_description = "Predicted Class"

    def confidence_display(self, obj: ClassificationHistory) -> str:
        return f"{obj.confidence_percentage:.2f}%"

    confidence_display.short_description = "Confidence"

    def classification_time_display(self, obj: ClassificationHistory) -> str:
        return f"{obj.classification_time:.3f}s"

    classification_time_display.short_description = "Time"


admin.site.site_header = "PREDICTOR (VisionClassify) Administration"
admin.site.site_title = "PREDICTOR Admin"
admin.site.index_title = "AI Image Classification Management"
