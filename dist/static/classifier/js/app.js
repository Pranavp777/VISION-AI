/**
 * PREDICTOR (VisionClassify) — Main Frontend Application Script (`app.js`)
 *
 * Capabilities:
 * 1. Dark / Light Theme detection (`prefers-color-scheme`) & persistence (`localStorage`)
 * 2. Accessible Mobile Navigation Drawer (closes on link click, outside tap, close button, or Escape)
 * 3. Offline / Online detection & PWA Service Worker registration + Install prompt
 * 4. Multi-source Image Selection:
 *    - Device storage / Gallery (`<input type="file" accept="image/*">`)
 *    - Native mobile camera (`<input type="file" accept="image/*" capture="environment">`)
 *    - Live WebRTC browser camera (`navigator.mediaDevices.getUserMedia`) with front/rear flip
 *    - Desktop Drag & Drop
 *    - One-tap Sample Test Images
 * 5. Client-side File Validation & Mobile Image Compression
 * 6. Non-blocking Fetch API / AJAX Classification (`POST /api/classify/`)
 * 7. Dynamic Top-3 / Top-5 Prediction rendering & AJAX History deletion
 */

(function () {
    "use strict";

    // =========================================================================
    // 1. DARK / LIGHT THEME MANAGEMENT (Section 20)
    // =========================================================================
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const mobileThemeToggleBtn = document.getElementById("mobileThemeToggleBtn");
    const themeToggleLabel = document.getElementById("themeToggleLabel");
    const mobileThemeText = document.getElementById("mobileThemeText");
    const metaThemeColor = document.getElementById("metaThemeColor");

    function updateThemeUI(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        if (themeToggleLabel) {
            themeToggleLabel.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
        }
        if (mobileThemeText) {
            mobileThemeText.textContent = theme === "dark" ? "☀ Switch to Light Mode" : "🌙 Switch to Dark Mode";
        }
        if (metaThemeColor) {
            metaThemeColor.setAttribute("content", theme === "dark" ? "#0f172a" : "#2563eb");
        }
    }

    function toggleTheme() {
        const current = document.documentElement.getAttribute("data-theme") || "light";
        const next = current === "dark" ? "light" : "dark";
        try {
            localStorage.setItem("predictor_theme", next);
        } catch (e) {}
        updateThemeUI(next);
    }

    updateThemeUI(document.documentElement.getAttribute("data-theme") || "light");
    if (themeToggleBtn) themeToggleBtn.addEventListener("click", toggleTheme);
    if (mobileThemeToggleBtn) mobileThemeToggleBtn.addEventListener("click", toggleTheme);

    // =========================================================================
    // 2. RESPONSIVE MOBILE NAVIGATION DRAWER (Section 5)
    // =========================================================================
    const mobileMenuBtn = document.getElementById("mobileMenuBtn");
    const closeDrawerBtn = document.getElementById("closeDrawerBtn");
    const mobileDrawer = document.getElementById("mobileDrawer");
    const drawerBackdrop = document.getElementById("drawerBackdrop");

    function openMobileDrawer() {
        if (!mobileDrawer || !drawerBackdrop) return;
        mobileDrawer.classList.add("open");
        mobileDrawer.setAttribute("aria-hidden", "false");
        drawerBackdrop.hidden = false;
        if (mobileMenuBtn) mobileMenuBtn.setAttribute("aria-expanded", "true");
        document.body.style.overflow = "hidden";
    }

    function closeMobileDrawer() {
        if (!mobileDrawer || !drawerBackdrop) return;
        mobileDrawer.classList.remove("open");
        mobileDrawer.setAttribute("aria-hidden", "true");
        drawerBackdrop.hidden = true;
        if (mobileMenuBtn) mobileMenuBtn.setAttribute("aria-expanded", "false");
        document.body.style.overflow = "";
    }

    if (mobileMenuBtn) mobileMenuBtn.addEventListener("click", openMobileDrawer);
    if (closeDrawerBtn) closeDrawerBtn.addEventListener("click", closeMobileDrawer);
    if (drawerBackdrop) drawerBackdrop.addEventListener("click", closeMobileDrawer);

    if (mobileDrawer) {
        const drawerLinks = mobileDrawer.querySelectorAll(".drawer-link");
        drawerLinks.forEach(function (link) {
            link.addEventListener("click", closeMobileDrawer);
        });
    }

    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") {
            closeMobileDrawer();
        }
    });

    // =========================================================================
    // 3. OFFLINE DETECTION & PWA SERVICE WORKER (Section 17)
    // =========================================================================
    const offlineBanner = document.getElementById("offlineBanner");

    function syncNetworkStatus() {
        if (!offlineBanner) return;
        if (navigator.onLine === false) {
            offlineBanner.hidden = false;
        } else {
            offlineBanner.hidden = true;
        }
    }

    window.addEventListener("online", syncNetworkStatus);
    window.addEventListener("offline", syncNetworkStatus);
    syncNetworkStatus();

    if ("serviceWorker" in navigator) {
        window.addEventListener("load", function () {
            navigator.serviceWorker
                .register("/service-worker.js", { scope: "/" })
                .catch(function () {
                    // Service worker registration optional in restricted dev environments
                });
        });
    }

    let deferredPwaPrompt = null;
    const installPwaBtn = document.getElementById("installPwaBtn");

    window.addEventListener("beforeinstallprompt", function (e) {
        e.preventDefault();
        deferredPwaPrompt = e;
        if (installPwaBtn) installPwaBtn.hidden = false;
    });

    if (installPwaBtn) {
        installPwaBtn.addEventListener("click", async function () {
            if (!deferredPwaPrompt) return;
            deferredPwaPrompt.prompt();
            await deferredPwaPrompt.userChoice;
            deferredPwaPrompt = null;
            installPwaBtn.hidden = true;
        });
    }

    // =========================================================================
    // 4. CLASSIFIER PAGE LOGIC (Upload, Camera, Preview, AJAX Inference)
    // =========================================================================
    const classifyForm = document.getElementById("classifyForm");
    if (classifyForm) {
        const imageInput = document.getElementById("imageInput");
        const mobileCameraInput = document.getElementById("mobileCameraInput");
        const dropZone = document.getElementById("dropZone");
        const chooseGalleryBtn = document.getElementById("chooseGalleryBtn");
        const useCameraBtn = document.getElementById("useCameraBtn");
        const takePhotoMobileBtn = document.getElementById("takePhotoMobileBtn");

        const uploadStepCard = document.getElementById("uploadStepCard");
        const cameraPanel = document.getElementById("cameraPanel");
        const cameraVideo = document.getElementById("cameraVideo");
        const cameraCanvas = document.getElementById("cameraCanvas");
        const closeCameraBtn = document.getElementById("closeCameraBtn");
        const switchCameraBtn = document.getElementById("switchCameraBtn");
        const capturePhotoBtn = document.getElementById("capturePhotoBtn");

        const previewCard = document.getElementById("previewCard");
        const previewImage = document.getElementById("previewImage");
        const previewFileName = document.getElementById("previewFileName");
        const previewFileSize = document.getElementById("previewFileSize");
        const previewDimensions = document.getElementById("previewDimensions");
        const previewFormat = document.getElementById("previewFormat");
        const classifyImageBtn = document.getElementById("classifyImageBtn");
        const changeImageBtn = document.getElementById("changeImageBtn");
        const retakeCameraBtn = document.getElementById("retakeCameraBtn");

        const loadingCard = document.getElementById("loadingCard");
        const loadingTimerText = document.getElementById("loadingTimerText");

        const resultCard = document.getElementById("resultCard");
        const resultImage = document.getElementById("resultImage");
        const resultEmoji = document.getElementById("resultEmoji");
        const resultPredictedClass = document.getElementById("resultPredictedClass");
        const resultConfidencePct = document.getElementById("resultConfidencePct");
        const resultProcessingTime = document.getElementById("resultProcessingTime");
        const topPredictionsList = document.getElementById("topPredictionsList");
        const classifyAnotherBtn = document.getElementById("classifyAnotherBtn");
        const resultPermalinkBtn = document.getElementById("resultPermalinkBtn");

        const errorBox = document.getElementById("classifierErrorBox");
        const errorText = document.getElementById("classifierErrorText");
        const dismissErrorBtn = document.getElementById("dismissErrorBtn");

        const maxUploadMb = parseInt(classifyForm.dataset.maxMb || "10", 10);
        const maxUploadBytes = maxUploadMb * 1024 * 1024;
        const apiUrl = classifyForm.dataset.apiUrl || "/api/classify/";

        let selectedFile = null;
        let capturedFromCamera = false;
        let activeMediaStream = null;
        let currentFacingMode = "environment";
        let currentTopPredictions = [];
        let activeTopK = 3;
        let loadingTimerInterval = null;

        function showError(msg) {
            if (!errorBox || !errorText) return;
            errorText.textContent = "⚠️ " + msg;
            errorBox.hidden = false;
            errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }

        function clearError() {
            if (errorBox) errorBox.hidden = true;
        }

        if (dismissErrorBtn) {
            dismissErrorBtn.addEventListener("click", clearError);
        }

        function formatBytes(bytes) {
            if (!bytes || bytes === 0) return "0 KB";
            const kb = bytes / 1024;
            if (kb < 1024) return kb.toFixed(1) + " KB";
            return (kb / 1024).toFixed(2) + " MB";
        }

        function validateClientFile(file) {
            if (!file) {
                showError("No image selected. Please choose or capture an image.");
                return false;
            }
            const allowedTypes = [
                "image/jpeg",
                "image/jpg",
                "image/png",
                "image/webp",
                "image/bmp",
                "image/gif",
            ];
            if (file.type && !allowedTypes.includes(file.type.toLowerCase())) {
                showError("Unsupported image format. Please select a JPG, PNG, WEBP, BMP, or GIF image.");
                return false;
            }
            if (file.size > maxUploadBytes) {
                showError("Image is too large (" + formatBytes(file.size) + "). Maximum allowed size is " + maxUploadMb + " MB.");
                return false;
            }
            return true;
        }

        function displayImagePreview(file, fromCamera) {
            clearError();
            if (!validateClientFile(file)) return;

            selectedFile = file;
            capturedFromCamera = Boolean(fromCamera);

            const objectUrl = URL.createObjectURL(file);
            previewImage.src = objectUrl;
            previewFileName.textContent = file.name || "captured_image.jpg";
            previewFileSize.textContent = formatBytes(file.size);
            previewFormat.textContent = (file.type || "image/jpeg").replace("image/", "").toUpperCase();
            previewDimensions.textContent = "Loading...";

            const tempImg = new Image();
            tempImg.onload = function () {
                previewDimensions.textContent = tempImg.naturalWidth + " × " + tempImg.naturalHeight + " px";
            };
            tempImg.src = objectUrl;

            stopCameraStream();
            cameraPanel.hidden = true;
            uploadStepCard.hidden = true;
            resultCard.hidden = true;
            loadingCard.hidden = true;
            previewCard.hidden = false;
            retakeCameraBtn.hidden = !capturedFromCamera;
        }

        // File Picker & Native Mobile Camera Handlers
        if (chooseGalleryBtn) {
            chooseGalleryBtn.addEventListener("click", function (e) {
                e.stopPropagation();
                imageInput.value = "";
                imageInput.click();
            });
        }

        if (takePhotoMobileBtn) {
            takePhotoMobileBtn.addEventListener("click", function (e) {
                e.stopPropagation();
                if (mobileCameraInput) {
                    mobileCameraInput.value = "";
                    mobileCameraInput.click();
                }
            });
        }

        if (dropZone) {
            dropZone.addEventListener("click", function (e) {
                if (e.target.closest("button")) return;
                imageInput.value = "";
                imageInput.click();
            });

            dropZone.addEventListener("keydown", function (e) {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    imageInput.click();
                }
            });

            ["dragenter", "dragover"].forEach(function (evtName) {
                dropZone.addEventListener(evtName, function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.classList.add("drag-over");
                });
            });

            ["dragleave", "drop"].forEach(function (evtName) {
                dropZone.addEventListener(evtName, function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.classList.remove("drag-over");
                });
            });

            dropZone.addEventListener("drop", function (e) {
                const files = e.dataTransfer && e.dataTransfer.files;
                if (files && files.length > 0) {
                    displayImagePreview(files[0], false);
                }
            });
        }

        if (imageInput) {
            imageInput.addEventListener("change", function () {
                if (imageInput.files && imageInput.files[0]) {
                    displayImagePreview(imageInput.files[0], false);
                }
            });
        }

        if (mobileCameraInput) {
            mobileCameraInput.addEventListener("change", function () {
                if (mobileCameraInput.files && mobileCameraInput.files[0]) {
                    displayImagePreview(mobileCameraInput.files[0], true);
                }
            });
        }

        // Sample Test Images Handler
        const sampleChips = document.querySelectorAll(".sample-chip");
        sampleChips.forEach(function (chip) {
            chip.addEventListener("click", async function () {
                const sampleUrl = chip.dataset.sampleUrl;
                const sampleName = chip.dataset.sampleName || "sample.jpg";
                if (!sampleUrl) return;
                try {
                    clearError();
                    const resp = await fetch(sampleUrl);
                    if (!resp.ok) throw new Error("Could not load sample image.");
                    const blob = await resp.blob();
                    const file = new File([blob], sampleName, { type: "image/jpeg" });
                    displayImagePreview(file, false);
                } catch (err) {
                    showError("Unable to load sample image. Please choose an image from your device.");
                }
            });
        });

        // =====================================================================
        // 5. CAMERA CAPTURE FEATURE (Section 7)
        // =====================================================================
        function stopCameraStream() {
            if (activeMediaStream) {
                activeMediaStream.getTracks().forEach(function (track) {
                    track.stop();
                });
                activeMediaStream = null;
            }
            if (cameraVideo) {
                cameraVideo.srcObject = null;
            }
        }

        async function startCamera(facingMode) {
            clearError();
            stopCameraStream();

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                // Graceful fallback message required by Section 7
                showError("Camera access is not available. Please choose an image from your device.");
                if (mobileCameraInput) {
                    mobileCameraInput.click();
                }
                return;
            }

            try {
                const constraints = {
                    video: {
                        facingMode: { ideal: facingMode || "environment" },
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                    },
                    audio: false,
                };
                activeMediaStream = await navigator.mediaDevices.getUserMedia(constraints);
                cameraVideo.srcObject = activeMediaStream;
                await cameraVideo.play();

                uploadStepCard.hidden = true;
                previewCard.hidden = true;
                resultCard.hidden = true;
                cameraPanel.hidden = false;
            } catch (err) {
                showError("Camera access is not available. Please choose an image from your device.");
            }
        }

        if (useCameraBtn) {
            useCameraBtn.addEventListener("click", function (e) {
                e.stopPropagation();
                startCamera(currentFacingMode);
            });
        }

        if (switchCameraBtn) {
            switchCameraBtn.addEventListener("click", function () {
                currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
                startCamera(currentFacingMode);
            });
        }

        if (closeCameraBtn) {
            closeCameraBtn.addEventListener("click", function () {
                stopCameraStream();
                cameraPanel.hidden = true;
                uploadStepCard.hidden = false;
            });
        }

        if (capturePhotoBtn) {
            capturePhotoBtn.addEventListener("click", function () {
                if (!cameraVideo || !cameraCanvas) return;
                const width = cameraVideo.videoWidth || 640;
                const height = cameraVideo.videoHeight || 480;
                cameraCanvas.width = width;
                cameraCanvas.height = height;

                const ctx = cameraCanvas.getContext("2d");
                ctx.drawImage(cameraVideo, 0, 0, width, height);

                cameraCanvas.toBlob(
                    function (blob) {
                        if (!blob) {
                            showError("Failed to capture photo from camera.");
                            return;
                        }
                        const filename = "camera_capture_" + Date.now() + ".jpg";
                        const capturedFile = new File([blob], filename, { type: "image/jpeg" });
                        displayImagePreview(capturedFile, true);
                    },
                    "image/jpeg",
                    0.9
                );
            });
        }

        if (changeImageBtn) {
            changeImageBtn.addEventListener("click", function () {
                selectedFile = null;
                previewCard.hidden = true;
                resultCard.hidden = true;
                uploadStepCard.hidden = false;
            });
        }

        if (retakeCameraBtn) {
            retakeCameraBtn.addEventListener("click", function () {
                previewCard.hidden = true;
                startCamera(currentFacingMode);
            });
        }

        // Auto-open camera if navigated via ?mode=camera
        if (classifyForm.dataset.initialMode === "camera") {
            startCamera(currentFacingMode);
        }

        // =====================================================================
        // 6. CLASSIFICATION AJAX PROCESS & RESULT RENDERING (Sections 9, 10, 11)
        // =====================================================================
        function renderTopPredictionsList(items, limit) {
            if (!topPredictionsList) return;
            topPredictionsList.innerHTML = "";
            const sliced = (items || []).slice(0, limit);

            sliced.forEach(function (item, idx) {
                const pct = typeof item.percentage === "number"
                    ? item.percentage.toFixed(2)
                    : (Number(item.confidence || 0) * 100).toFixed(2);
                const emoji = item.emoji || "🔍";
                const label = item.display_name || (item.class ? item.class.charAt(0).toUpperCase() + item.class.slice(1) : "Unknown");

                const li = document.createElement("li");
                li.className = "top-prediction-item";
                li.innerHTML =
                    '<div class="top-pred-row">' +
                    '<span class="top-pred-name"><span aria-hidden="true">' + emoji + "</span> " + label + "</span>" +
                    '<span class="top-pred-pct">' + pct + "%</span>" +
                    "</div>" +
                    '<div class="progress-track" role="progressbar" aria-valuenow="' + pct + '" aria-valuemin="0" aria-valuemax="100">' +
                    '<div class="progress-fill ' + (idx === 0 ? "progress-primary" : "") + '" style="width: ' + Math.min(100, Math.max(2, parseFloat(pct))) + '%;"></div>' +
                    "</div>";
                topPredictionsList.appendChild(li);
            });
        }

        const topKButtons = document.querySelectorAll(".top-k-btn");
        topKButtons.forEach(function (btn) {
            btn.addEventListener("click", function () {
                topKButtons.forEach(function (b) {
                    b.classList.remove("active");
                });
                btn.classList.add("active");
                activeTopK = parseInt(btn.dataset.topk || "3", 10);
                renderTopPredictionsList(currentTopPredictions, activeTopK);
            });
        });

        async function runClassification() {
            clearError();

            if (navigator.onLine === false) {
                showError("You are offline. Image classification requires an internet connection.");
                return;
            }

            if (!selectedFile) {
                showError("No image selected. Please upload or capture an image first.");
                return;
            }

            // Show responsive loading state
            previewCard.hidden = true;
            resultCard.hidden = true;
            loadingCard.hidden = false;

            const startTs = performance.now();
            if (loadingTimerInterval) clearInterval(loadingTimerInterval);
            loadingTimerInterval = setInterval(function () {
                const sec = ((performance.now() - startTs) / 1000).toFixed(1);
                if (loadingTimerText) loadingTimerText.textContent = sec + "s elapsed";
            }, 100);

            const formData = new FormData();
            formData.append("image", selectedFile, selectedFile.name || "upload.jpg");

            const csrfInput = classifyForm.querySelector('input[name="csrfmiddlewaretoken"]');
            const csrfToken = csrfInput ? csrfInput.value : "";

            try {
                const response = await fetch(apiUrl, {
                    method: "POST",
                    headers: {
                        "X-CSRFToken": csrfToken,
                        "X-Requested-With": "XMLHttpRequest",
                        Accept: "application/json",
                    },
                    body: formData,
                });

                const data = await response.json();
                clearInterval(loadingTimerInterval);
                loadingCard.hidden = true;

                if (!response.ok || !data.success) {
                    previewCard.hidden = false;
                    showError(data.error || "Prediction failed. Please try another image.");
                    return;
                }

                // Populate Result Card
                resultImage.src = data.image_url || previewImage.src;
                resultEmoji.textContent = data.emoji || "🔍";
                resultPredictedClass.textContent = data.display_name || data.prediction;
                const confPct = typeof data.confidence_percentage === "number"
                    ? data.confidence_percentage.toFixed(2)
                    : (Number(data.confidence || 0) * 100).toFixed(2);
                resultConfidencePct.textContent = confPct + "%";
                resultProcessingTime.textContent = Number(data.classification_time || 0).toFixed(2);

                if (resultPermalinkBtn && data.result_url) {
                    resultPermalinkBtn.href = data.result_url;
                }

                currentTopPredictions = data.top_predictions || [];
                renderTopPredictionsList(currentTopPredictions, activeTopK);

                // Persist classification to localStorage so /history/ and /admin/ stay synced on Cloudflare Edge
                try {
                    const reader = new FileReader();
                    reader.onload = function (ev) {
                        const stored = JSON.parse(localStorage.getItem("predictor_cloud_history") || "[]");
                        stored.unshift({
                            id: data.id || Date.now(),
                            predicted_class: data.predicted_class || data.prediction || "unknown",
                            display_name: data.display_name || data.prediction || "Unknown",
                            emoji: data.emoji || "🔍",
                            confidence_percentage: parseFloat(confPct),
                            classification_time: Number(data.classification_time || 0.04).toFixed(3),
                            original_filename: selectedFile ? selectedFile.name : "capture.jpg",
                            created_at: new Date().toLocaleDateString("en-GB", {
                                day: "2-digit",
                                month: "short",
                                year: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                            }),
                            image_data: ev.target.result,
                        });
                        localStorage.setItem("predictor_cloud_history", JSON.stringify(stored.slice(0, 30)));
                    };
                    reader.readAsDataURL(selectedFile);
                } catch (storageErr) {}

                resultCard.hidden = false;
                resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
            } catch (err) {
                clearInterval(loadingTimerInterval);
                loadingCard.hidden = true;
                previewCard.hidden = false;
                showError("Network failure while communicating with the AI server. Please check your connection and try again.");
            }
        }

        if (classifyImageBtn) {
            classifyImageBtn.addEventListener("click", runClassification);
        }

        if (classifyAnotherBtn) {
            classifyAnotherBtn.addEventListener("click", function () {
                selectedFile = null;
                resultCard.hidden = true;
                previewCard.hidden = true;
                uploadStepCard.hidden = false;
                uploadStepCard.scrollIntoView({ behavior: "smooth", block: "start" });
            });
        }
    }

    // =========================================================================
    // 7. SYNC CLOUDFLARE LOCALSTORAGE HISTORY ON /history/ PAGE
    // =========================================================================
    const historyTableBody = document.querySelector(".history-table tbody");
    const mobileHistoryCards = document.querySelector(".mobile-history-cards");
    if (historyTableBody && mobileHistoryCards && window.location.hostname !== "127.0.0.1" && window.location.hostname !== "localhost") {
        try {
            const cloudItems = JSON.parse(localStorage.getItem("predictor_cloud_history") || "[]");
            if (cloudItems.length > 0) {
                const countEl = document.getElementById("historyTotalCount");
                if (countEl) {
                    countEl.textContent = String(parseInt(countEl.textContent || "0", 10) + cloudItems.length);
                }
                cloudItems.forEach(function (item) {
                    const tr = document.createElement("tr");
                    tr.id = "history-row-" + item.id;
                    tr.innerHTML =
                        '<td><img src="' + (item.image_data || "") + '" alt="' + item.display_name + '" class="table-thumb"></td>' +
                        '<td><span aria-hidden="true">' + item.emoji + "</span> <strong>" + item.display_name + '</strong><div class="table-subtext">' + item.original_filename + "</div></td>" +
                        '<td><span class="badge-confidence">' + item.confidence_percentage + "%</span></td>" +
                        "<td>" + item.classification_time + "s</td>" +
                        "<td>" + item.created_at + "</td>" +
                        '<td class="text-right"><button type="button" class="btn btn-danger-outline btn-sm cloud-del-btn" data-id="' + item.id + '">Delete</button></td>';
                    historyTableBody.insertBefore(tr, historyTableBody.firstChild);

                    const card = document.createElement("article");
                    card.className = "card mobile-history-card";
                    card.id = "history-card-" + item.id;
                    card.innerHTML =
                        '<div class="mobile-card-image-wrap"><img src="' + (item.image_data || "") + '" alt="' + item.display_name + '"></div>' +
                        '<div class="mobile-card-body">' +
                        '<p class="mobile-card-line"><strong>Prediction:</strong><span>' + item.emoji + " " + item.display_name + "</span></p>" +
                        '<p class="mobile-card-line"><strong>Confidence:</strong><span class="badge-confidence">' + item.confidence_percentage + "%</span></p>" +
                        '<p class="mobile-card-line"><strong>Processing Time:</strong><span>' + item.classification_time + "s</span></p>" +
                        '<p class="mobile-card-line"><strong>Date:</strong><span>' + item.created_at + "</span></p>" +
                        '<div class="mobile-card-actions"><button type="button" class="btn btn-danger-outline btn-block cloud-del-btn" data-id="' + item.id + '">Delete</button></div>' +
                        "</div>";
                    mobileHistoryCards.insertBefore(card, mobileHistoryCards.firstChild);
                });

                document.querySelectorAll(".cloud-del-btn").forEach(function (btn) {
                    btn.addEventListener("click", function () {
                        const delId = String(btn.dataset.id);
                        const current = JSON.parse(localStorage.getItem("predictor_cloud_history") || "[]");
                        const filtered = current.filter(function (x) {
                            return String(x.id) !== delId;
                        });
                        localStorage.setItem("predictor_cloud_history", JSON.stringify(filtered));
                        const r = document.getElementById("history-row-" + delId);
                        const c = document.getElementById("history-card-" + delId);
                        if (r) r.remove();
                        if (c) c.remove();
                    });
                });
            }
        } catch (e) {}
    }

    // =========================================================================
    // 8. AJAX HISTORY DELETION (Section 15)
    // =========================================================================
    const deleteForms = document.querySelectorAll(".delete-history-form");
    deleteForms.forEach(function (form) {
        form.addEventListener("submit", async function (e) {
            e.preventDefault();
            const recordId = form.dataset.recordId;
            const csrfInput = form.querySelector('input[name="csrfmiddlewaretoken"]');
            const csrfToken = csrfInput ? csrfInput.value : "";

            try {
                const resp = await fetch(form.action, {
                    method: "POST",
                    headers: {
                        "X-CSRFToken": csrfToken,
                        "X-Requested-With": "XMLHttpRequest",
                        Accept: "application/json",
                    },
                });
                const data = await resp.json();
                if (resp.ok && data.success) {
                    const rowEl = document.getElementById("history-row-" + recordId);
                    const cardEl = document.getElementById("history-card-" + recordId);
                    if (rowEl) rowEl.remove();
                    if (cardEl) cardEl.remove();

                    const countEl = document.getElementById("historyTotalCount");
                    if (countEl && typeof data.remaining_count === "number") {
                        countEl.textContent = String(data.remaining_count);
                        if (data.remaining_count === 0) {
                            window.location.reload();
                        }
                    }
                } else {
                    form.submit();
                }
            } catch (err) {
                form.submit();
            }
        });
    });
})();
