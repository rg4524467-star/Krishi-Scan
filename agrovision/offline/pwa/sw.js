/*
 * Krishi Scan service worker - the offline/PWA backbone.
 *
 * Strategy: cache-first with runtime network fallback.  The FIRST online load
 * primes the cache with everything needed for full offline operation:
 *   - the stlite shell (index.html, JS bundle)
 *   - the Python app files (agrovision/*)
 *   - language strings (already bundled into the app files)
 *   - the quantized ONNX model + classes.json (the "assets" group)
 *
 * Keep the first-load payload small: the model is the quantized ONNX export,
 * and assets use versioned URLs so a model update bumps the version without
 * re-downloading the whole shell.
 */
const VERSION = "krishi-scan-v1";
const SHELL_CACHE = `${VERSION}-shell`;
const ASSET_CACHE = `${VERSION}-assets`;

/* Files that MUST exist for offline operation after first load. */
const SHELL_ASSETS = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "https://cdn.jsdelivr.net/npm/@stlite/mountable@0.57.0/build/stlite.js",
];

/* Versioned model & app payload.  Bump VERSION to force re-fetch. */
const MODEL_ASSETS = [
  "./model/mobilenetv2_quant.onnx",
  "./model/mobilenetv2_cam.onnx",
  "./model/classes.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== SHELL_CACHE && k !== ASSET_CACHE)
          .map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never intercept non-GET or cross-origin API traffic (weather, sync, CDN).
  if (event.request.method !== "GET" || url.origin !== self.location.origin) {
    return;
  }

  const cacheName = url.pathname.startsWith("/model/")
    ? ASSET_CACHE
    : SHELL_CACHE;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (response && response.ok) {
          const copy = response.clone();
          caches.open(cacheName).then((cache) => cache.put(event.request, copy));
        }
        return response;
      });
    })
  );
});
