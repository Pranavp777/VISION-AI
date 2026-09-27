# PREDICTOR (VisionClassify) — AI Image Classification Web Application

**PREDICTOR (`VisionClassify`)** is a complete, production-ready, mobile-first **Computer Vision Image Classification** web platform built with **Python 3.11+**, **Django**, **Django REST Framework**, **TensorFlow/Keras**, **OpenCV**, **Pillow**, **HTML5**, **CSS3**, and **Vanilla JavaScript**.

---

## 1. Key Features

- **Universal Mobile-First Responsive Design**: Adapts seamlessly from `320px` small phones up to `1440px+` desktop monitors using CSS Grid, Flexbox, and fluid typography (`clamp()`). Zero horizontal overflow and `44px × 44px` minimum touch targets.
- **Multi-Source Image Input**:
  - Upload from device storage, photo gallery, or file manager (`<input type="file" accept="image/*">`).
  - Direct Mobile Camera capture (`capture="environment"`).
  - Live browser WebRTC Camera viewfinder (`navigator.mediaDevices.getUserMedia`) with front/rear camera flip and graceful permission-denial fallback.
  - Desktop Drag-and-Drop upload zone.
  - One-tap synthetic sample test images (`Cat`, `Dog`, `Car`, `Airplane`, `Flower`).
- **Singleton TensorFlow/Keras Inference (`ml/predictor.py`)**:
  - Loads `ml/model.keras` and `ml/class_names.json` **only once** per worker process.
  - Uses **Pillow** for EXIF orientation correction and binary header validation, and **OpenCV** (`cv2`) for resizing and normalization.
  - Returns top predicted class, confidence score, Top 3 / Top 5 predictions, and processing time.
- **Responsive History Management**:
  - Renders a full data table on Desktop (`>= 768px`) and automatically transforms into touch-friendly vertical cards on Mobile (`< 768px`).
- **Progressive Web App (PWA)**:
  - Includes `manifest.json` and root-scoped `service-worker.js` for static UI offline caching and "Add to Home Screen".
  - Displays `"You are offline. Image classification requires an internet connection."` when offline.
- **Dark / Light Mode (`☀ / 🌙`)**:
  - Automatically detects OS color-scheme preference initially and saves manual toggles in `localStorage`.
- **REST API (`POST /api/classify/`)**:
  - Accepts `multipart/form-data` with `image=<file>` and returns structured JSON predictions.

---

## 2. Project Structure

```text
VisionClassify/
│
├── manage.py
├── requirements.txt
├── README.md
├── .env.example
├── Procfile
├── render.yaml
├── runtime.txt
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── classifier/
│   ├── __init__.py
│   ├── apps.py
│   ├── migrations/
│   │   ├── __init__.py
│   │   └── 0001_initial.py
│   ├── templates/
│   │   └── classifier/
│   │       ├── base.html
│   │       ├── home.html
│   │       ├── classifier.html
│   │       ├── result.html
│   │       ├── history.html
│   │       └── about.html
│   ├── static/
│   │   └── classifier/
│   │       ├── css/
│   │       │   └── style.css
│   │       ├── js/
│   │       │   └── app.js
│   │       ├── icons/
│   │       │   ├── favicon.svg
│   │       │   ├── icon-192.png
│   │       │   └── icon-512.png
│   │       ├── manifest.json
│   │       └── service-worker.js
│   ├── models.py
│   ├── views.py
│   ├── forms.py
│   ├── urls.py
│   ├── admin.py
│   ├── tests.py
│   └── ml_model.py
│
├── ml/
│   ├── __init__.py
│   ├── model.keras
│   ├── class_names.json
│   ├── predictor.py
│   └── train.py
│
├── media/
│   └── uploads/
│
└── static/
```

---

## 3. Windows Development Setup Commands

Run the following commands in **PowerShell** or **Command Prompt** from the project root (`d:\VISION AI`):

```powershell
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

python ml/train.py --mode bootstrap

python manage.py makemigrations

python manage.py migrate

python manage.py createsuperuser

python manage.py runserver
```

Open your browser at:
**http://127.0.0.1:8000/**

---

## 4. Training / Updating the Machine Learning Model

- **Instant Bootstrap Model (Default)**:
  ```powershell
  python ml/train.py --mode bootstrap
  ```
- **Train on CIFAR-10 Dataset**:
  ```powershell
  python ml/train.py --mode cifar10 --epochs 10
  ```
- **Train on Custom Image Folder (`dataset/<class_name>/*.jpg`)**:
  ```powershell
  python ml/train.py --mode custom --data-dir ./dataset --epochs 15
  ```

Class names are stored dynamically in `ml/class_names.json` and are never hard-coded.

---

## 5. REST API Documentation

### Endpoint
`POST /api/classify/`

### Request (`multipart/form-data`)
```bash
curl -X POST http://127.0.0.1:8000/api/classify/ \
  -F "image=@path/to/photo.jpg"
```

### Response (`200 OK`)
```json
{
    "success": true,
    "prediction": "cat",
    "confidence": 0.9642,
    "top_predictions": [
        {
            "class": "cat",
            "confidence": 0.9642
        }
    ]
}
```

---

## 6. Device & Viewport Testing Matrix

Tested and verified across all required viewports:
- `320 × 568` (iPhone SE / Small Mobile)
- `375 × 667` (iPhone 8 / Standard Mobile)
- `390 × 844` (iPhone 14 / Modern Mobile)
- `412 × 915` (Android Pixel / Galaxy)
- `768 × 1024` (iPad Portrait / Tablet)
- `1024 × 768` (iPad Landscape / Small Laptop)
- `1280 × 720` (HD Laptop)
- `1366 × 768` (Widescreen Laptop)
- `1440 × 900` (Desktop Monitor)
- `1920 × 1080` (Full HD Desktop)

---

## 7. Production Deployment Instructions

### Environment Variables
Set the following environment variables in your production host:
- `DJANGO_SECRET_KEY`: Strong random 50+ character key
- `DJANGO_DEBUG`: `False`
- `DJANGO_ALLOWED_HOSTS`: `yourdomain.com,www.yourdomain.com`
- `DJANGO_CSRF_TRUSTED_ORIGINS`: `https://yourdomain.com`
- `DATABASE_URL`: `postgres://user:password@host:5432/dbname`

### A. Render / Railway
1. Push repository to GitHub.
2. Connect the repository to **Render** (uses `render.yaml` automatically) or **Railway** (uses `Procfile`).
3. Attach a managed **PostgreSQL** database (`DATABASE_URL` is injected automatically).
4. Build command:
   ```bash
   pip install -r requirements.txt && python ml/train.py --mode bootstrap && python manage.py collectstatic --noinput && python manage.py migrate --noinput
   ```
5. Start command:
   ```bash
   gunicorn config.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120
   ```

### B. AWS / Google Cloud / Azure / Linux VPS
1. Provision Ubuntu 22.04+ server with Python 3.11+, PostgreSQL, and Nginx.
2. Clone repository, create `.venv`, and run `pip install -r requirements.txt`.
3. Generate model and static assets:
   ```bash
   python ml/train.py --mode bootstrap
   python manage.py migrate --noinput
   python manage.py collectstatic --noinput
   ```
4. Run Gunicorn behind Nginx systemd service:
   ```bash
   gunicorn config.wsgi:application --bind 127.0.0.1:8000 --workers 3 --timeout 120
   ```

### C. PythonAnywhere
1. Upload code or clone via Bash console.
2. Create virtualenv (`mkvirtualenv --python=/usr/bin/python3.11 predictor-venv`) and `pip install -r requirements.txt`.
3. Configure WSGI file to point to `config.wsgi.application` and map `/static/` to `static/` and `/media/` to `media/`.
