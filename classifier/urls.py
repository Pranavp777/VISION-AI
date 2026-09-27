"""
URL routing for the `classifier` app in VisionClassify (PREDICTOR).
"""

from django.urls import path

from classifier import views

app_name = "classifier"

urlpatterns = [
    # Main pages
    path("", views.home_view, name="home"),
    path("classifier/", views.classifier_view, name="classifier"),
    path("result/<int:pk>/", views.result_view, name="result"),
    path("history/", views.history_view, name="history"),
    path("history/delete/<int:pk>/", views.delete_history_view, name="delete_history"),
    path("history/clear/", views.clear_history_view, name="clear_history"),
    path("about/", views.about_view, name="about"),
    # REST API
    path("api/classify/", views.ClassifyImageAPIView.as_view(), name="api_classify"),
    path("api/sample-image/<str:category>/", views.sample_image_view, name="sample_image"),
    # Root-level PWA assets
    path("manifest.json", views.manifest_view, name="manifest"),
    path("service-worker.js", views.service_worker_view, name="service_worker"),
]
