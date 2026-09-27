"""App configuration for the classifier application."""

from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "classifier"
    verbose_name = "PREDICTOR — VisionClassify AI"
