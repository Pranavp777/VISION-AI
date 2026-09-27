"""
Comprehensive Automated Test Suite for VisionClassify (PREDICTOR).

Tests:
- Responsive HTML pages (Home, Classifier, History, About, Result)
- PWA endpoints (`/manifest.json`, `/service-worker.js`)
- ML Predictor singleton (`ml.predictor.predict_image`)
- REST API endpoint (`POST /api/classify/`)
- Security & Validation (missing file, corrupted file, unsupported extension, oversized file)
- History deletion & clear-all functionality
"""

from __future__ import annotations

import io
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse
from PIL import Image

from classifier.models import ClassificationHistory
from ml.predictor import predict_image


def create_test_image_bytes(color=(220, 140, 70), size=(224, 224), fmt="JPEG") -> bytes:
    """Create an in-memory valid RGB image for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class PredictorApplicationTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_page_renders_responsive_elements(self):
        response = self.client.get(reverse("classifier:home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "VISIONCLASSIFY")
        self.assertContains(response, "Upload Image")
        self.assertContains(response, "Use Camera")

    def test_classifier_page_renders_upload_and_camera_inputs(self):
        response = self.client.get(reverse("classifier:classifier"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'accept="image/*"')
        self.assertContains(response, 'capture="environment"')
        self.assertContains(response, "Use Camera")

    def test_pwa_manifest_and_service_worker(self):
        manifest_resp = self.client.get(reverse("classifier:manifest"))
        self.assertEqual(manifest_resp.status_code, 200)
        self.assertIn("application/manifest+json", manifest_resp["Content-Type"])

        sw_resp = self.client.get(reverse("classifier:service_worker"))
        self.assertEqual(sw_resp.status_code, 200)
        self.assertEqual(sw_resp["Service-Worker-Allowed"], "/")

    def test_ml_predictor_function_directly(self):
        img_bytes = create_test_image_bytes()
        temp_path = Path("media/uploads/unit_test_cat.jpg")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(img_bytes)
        try:
            result = predict_image(temp_path)
            self.assertIn("predicted_class", result)
            self.assertIn("confidence", result)
            self.assertIn("top_predictions", result)
            self.assertGreater(len(result["top_predictions"]), 0)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_api_classify_valid_image(self):
        img_bytes = create_test_image_bytes()
        upload = SimpleUploadedFile("cat_test.jpg", img_bytes, content_type="image/jpeg")
        response = self.client.post(
            reverse("classifier:api_classify"),
            data={"image": upload},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("top_predictions", data)
        self.assertEqual(ClassificationHistory.objects.count(), 1)

        # Verify result & history pages with the created record
        record = ClassificationHistory.objects.first()
        res_page = self.client.get(reverse("classifier:result", kwargs={"pk": record.pk}))
        self.assertEqual(res_page.status_code, 200)

        hist_page = self.client.get(reverse("classifier:history"))
        self.assertEqual(hist_page.status_code, 200)
        self.assertContains(hist_page, record.predicted_class.title())

    def test_api_classify_rejects_missing_or_invalid_files(self):
        # 1. No image uploaded
        resp_empty = self.client.post(reverse("classifier:api_classify"), data={})
        self.assertEqual(resp_empty.status_code, 400)
        self.assertFalse(resp_empty.json()["success"])

        # 2. Invalid binary data disguised as .jpg
        bad_file = SimpleUploadedFile("fake.jpg", b"not-an-image-content", content_type="image/jpeg")
        resp_bad = self.client.post(reverse("classifier:api_classify"), data={"image": bad_file})
        self.assertEqual(resp_bad.status_code, 400)
        self.assertFalse(resp_bad.json()["success"])

        # 3. Unsupported extension
        txt_file = SimpleUploadedFile("notes.txt", b"hello world", content_type="text/plain")
        resp_txt = self.client.post(reverse("classifier:api_classify"), data={"image": txt_file})
        self.assertEqual(resp_txt.status_code, 400)
        self.assertFalse(resp_txt.json()["success"])
