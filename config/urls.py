"""
Root URL configuration for VisionClassify (PREDICTOR) project.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path, re_path
from django.views.static import serve

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("classifier.urls")),
]

# Serve uploaded media files in both development and self-hosted container deployments
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
else:
    urlpatterns += [
        re_path(
            r"^media/(?P<path>.*)$",
            serve,
            {"document_root": settings.MEDIA_ROOT},
        ),
    ]

# Custom error handlers
handler400 = "classifier.views.error_400_view"
handler403 = "classifier.views.error_403_view"
handler404 = "classifier.views.error_404_view"
handler500 = "classifier.views.error_500_view"
