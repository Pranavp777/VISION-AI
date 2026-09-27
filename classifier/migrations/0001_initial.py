# Generated initial migration for ClassificationHistory

import classifier.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ClassificationHistory",
            fields=[
                ("id", models.BigAutoField(primary_key=True, serialize=False)),
                (
                    "image",
                    models.ImageField(
                        help_text="Uploaded or camera-captured image file.",
                        upload_to=classifier.models.safe_upload_path,
                    ),
                ),
                (
                    "original_filename",
                    models.CharField(
                        blank=True,
                        default="",
                        help_text="Sanitized display name of the uploaded file.",
                        max_length=255,
                    ),
                ),
                (
                    "predicted_class",
                    models.CharField(
                        db_index=True,
                        help_text="Top predicted class label.",
                        max_length=120,
                    ),
                ),
                (
                    "confidence",
                    models.FloatField(
                        help_text="Confidence score between 0.0 and 1.0 (e.g., 0.9642)."
                    ),
                ),
                (
                    "classification_time",
                    models.FloatField(
                        default=0.0,
                        help_text="Inference processing time in seconds.",
                    ),
                ),
                (
                    "top_predictions",
                    models.JSONField(
                        blank=True,
                        default=list,
                        help_text="List of top-K prediction dictionaries ({class, confidence, percentage}).",
                    ),
                ),
                (
                    "created_at",
                    models.DateTimeField(
                        auto_now_add=True,
                        db_index=True,
                        help_text="Timestamp when the classification was performed.",
                    ),
                ),
            ],
            options={
                "verbose_name": "Classification History",
                "verbose_name_plural": "Classification Histories",
                "ordering": ["-created_at"],
            },
        ),
    ]
