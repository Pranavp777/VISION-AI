/**
 * PREDICTOR (VisionClassify) — Progressive Web App Service Worker (`service-worker.js`)
 *
 * Responsibilities:
 * - Cache core static UI shell (HTML pages, CSS, JS, icons, manifest)
 * - Serve cached static assets when offline
 * - Intercept POST /api/classify/ when offline and return the exact required message:
 *   "You are offline. Image classification requires an internet connection."
 */

const CACHE_NAME = "predictor-visionclassify-v1";
const STATIC_ASSETS = [
    "/",
    "/classifier/",
    "/history/",
    "/about/",
    "/manifest.json",
    "/static/classifier/css/style.css",
    "/static/classifier/js/app.js",
    "/static/classifier/icons/favicon.svg",
    "/static/classifier/icons/icon-192.png",
    "/static/classifier/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
    event.waitUntil(
        caches
            .open(CACHE_NAME)
            .then((cache) => cache.addAll(STATIC_ASSETS))
            .then(() => self.skipWaiting())
    );
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches
            .keys()
            .then((keys) =>
                Promise.all(
                    keys
                        .filter((key) => key !== CACHE_NAME)
                        .map((oldKey) => caches.delete(oldKey))
                )
            )
            .then(() => self.clients.claim())
    );
});

self.addEventListener("fetch", (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Handle offline classification requests gracefully (Section 17)
    if (request.method === "POST" && url.pathname.includes("/classify")) {
        event.respondWith(
            fetch(request).catch(() =>
                new Response(
                    JSON.stringify({
                        success: false,
                        error: "You are offline. Image classification requires an internet connection.",
                    }),
                    {
                        status: 503,
                        headers: { "Content-Type": "application/json" },
                    }
                )
            )
        );
        return;
    }

    if (request.method !== "GET") {
        return;
    }

    // Network-first for HTML pages, falling back to cache when offline
    if (request.headers.get("Accept")?.includes("text/html")) {
        event.respondWith(
            fetch(request)
                .then((response) => {
                    const copy = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
                    return response;
                })
                .catch(() =>
                    caches.match(request).then((cached) => cached || caches.match("/"))
                )
        );
        return;
    }

    // Cache-first for static assets (CSS, JS, Icons)
    event.respondWith(
        caches.match(request).then((cachedResponse) => {
            if (cachedResponse) {
                return cachedResponse;
            }
            return fetch(request).then((networkResponse) => {
                if (
                    networkResponse &&
                    networkResponse.status === 200 &&
                    url.pathname.startsWith("/static/")
                ) {
                    const copy = networkResponse.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
                }
                return networkResponse;
            });
        })
    );
});
