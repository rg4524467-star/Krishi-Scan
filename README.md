# Krishi Scan 🌾

Crop disease detection for Indian farmers — **smartphone/PWA-first, offline-first**.

A Streamlit + stlite application where every module follows one rule:
**never let unverified input silently produce output the user trusts blindly.**
A blurry photo is caught by the *gatekeeper* before diagnosis, OCR'd soil
values are *shown back for one-tap confirm*, and every diagnosis comes with a
*visual explanation* (Grad-CAM/CAM heatmap), not just a label.

Languages: **हिन्दी · English · தமிழ்** (TTS readout included; no speech input).

---

## Feature summary

| Module | What it does | Online | Offline (stlite/Pyodide) |
|---|---|---|---|
| **CV Gatekeeper** | Blur (Laplacian var ≥100), brightness (40–220), leaf coverage (≥15% green), glare (≤5% near-white). Capture → check → feedback → retake loop. | OpenCV | Pure-NumPy fallback (same math) |
| **Classification** | MobileNetV2 (fine-tuned), **top-3** probabilities + **Grad-CAM heatmap**. | TensorFlow | Quantized ONNX + weights-based CAM |
| **Soil (optional)** | Manual entry **or** SHC photo → deskew/adaptive-threshold → OCR → **confirm/edit before use**. Skippable. | Tesseract/EasyOCR | Degrades to guided manual entry (no OCR engine in browser) |
| **Weather (optional)** | Current + last-3-days history (Open-Meteo). Skippable; failures never block. | HTTP API | Not available → photo-only diagnosis |
| **Fusion** | Hand-authored env ranges re-weight classifier probs (scikit-learn scoring). No multimodal training. | sklearn | sklearn (in Pyodide) |
| **Multilingual + TTS** | Externalized strings, Hindi/Tamil/English, TTS output. | browser/desktop TTS | browser TTS (speechSynthesis) |
| **Data** | SQLAlchemy: `Farmer`, `Diagnosis`, `SoilRecord`, `SyncQueue`. | central DB | JSON journal + auto-sync on reconnect |
| **Recommendations** | Treatment advice + next steps (agri dealer / Kisan Call Centre 1800-180-1551), offline-cached. | ✓ | ✓ (bundled in classes.json) |

Edge cases handled: 3+ retakes → *proceed anyway* with a low-confidence warning;
weather API failure → continue photo-only; out-of-distribution image → *"not
confident / possibly outside known conditions"* (no forced top-1); no internet
on first open → clear one-time-setup message; missing translation → falls back
to English, never a raw key.

---

## Project layout

```
AgroVision/
├── app.py                      # Streamlit entry (single file = stlite friendly)
├── requirements.txt
├── agrovision/
│   ├── config.py               # ALL thresholds, paths, env-overridable knobs
│   ├── gatekeeper/             # backends.py (cv2/numpy parity) checks.py gatekeeper.py
│   ├── inference/              # base.py tf_backend.py onnx_backend.py demo_backend.py
│   │                           # gradcam.py (TF Grad-CAM + ONNX CAM) classes.py
│   ├── soil/                   # preprocess.py ocr.py parser.py soil.py (confirm flow)
│   ├── weather/                # weather.py (Open-Meteo client)
│   ├── fusion/                 # reference.py (ranges table) scorer.py (sklearn)
│   ├── data/                   # models.py db.py offline_store.py sync.py
│   ├── lang/                   # strings_{en,hi,ta}.json translator.py tts.py
│   ├── reco/                   # recommendations.py (always a next-step)
│   ├── ui/                     # components.py styles.py state.py home/diagnose/history...
│   ├── offline/pwa/            # index.html (stlite shell) sw.js manifest.webmanifest
│   ├── assets/model/           # classes.json (class + multilingual content)
│   ├── assets/recommendations/ # conditions.json (env ranges for fusion)
│   └── util/                   # runtime.py image.py
├── scripts/
│   ├── finetune.py             # MobileNetV2 fine-tuning on merged dataset
│   └── export_onnx.py          # TF -> quantized ONNX + CAM subgraph
└── tests/                      # pytest (36 tests)
```

---

## Quick start — online (cloud) mode

```bash
cd AgroVision
python -m venv .venv && .venv\Scripts\activate      # Windows; or source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the printed URL on your phone (same Wi-Fi) for the mobile view. Without a
fine-tuned checkpoint the app runs in **demo mode** (`AGRO_DEMO=1`, clearly
labelled in the UI) so you can walk the whole flow.

> **Theme:** Krishi Scan is **light-theme only** by design. `.streamlit/config.toml`
> pins `base = "light"` and the settings toolbar is hidden, so the UI renders a
> consistent light palette regardless of the phone/OS dark-mode setting.

### Environment variables (all optional)

| Var | Purpose | Default |
|---|---|---|
| `AGRO_LANGUAGE` | default language `hi`/`en`/`ta` | `hi` |
| `AGRO_CHECKPOINT` | path to fine-tuned `.h5` model | empty (demo) |
| `AGRO_DEMO` | force demo backend | `0` |
| `AGRO_DB_URL` | SQLAlchemy URL (Postgres in prod) | `sqlite:///agrovision.db` |
| `AGRO_BLUR_MIN_VAR`, `AGRO_BRIGHT_MIN/MAX`, `AGRO_LEAF_MIN_COVER`, `AGRO_GLARE_MAX_RATIO` | gatekeeper thresholds | 100 / 40 / 220 / 0.15 / 0.05 |
| `AGRO_OCR` | `tesseract` or `easyocr` | `tesseract` |
| `AGRO_WEATHER_PROVIDER` | weather provider | `open-meteo` |

> **Tesseract system dependency (soil OCR, online only):**
> `pytesseract` is a Python wrapper — install the Tesseract binary too
> (e.g. `winget install UB-Mannheim.TesseractOCR`, or `apt install tesseract-ocr`
> on Linux) and set `TESSERACT_CMD` if needed.

---

## Offline (stlite / PWA) mode

The offline runtime is **stlite** — the same Streamlit app compiled to WebAssembly
via Pyodide. Build once, then install on a phone as a PWA.

```bash
# 1. Install the build tooling
pip install stlite

# 2. Package the app
stlite build --source app.py --out dist

# 3. Serve (any static server works; SW caching kicks in on first visit)
python -m http.server 8080 --directory dist
```

Open `http://localhost:8080` on your phone → **Add to Home Screen**. The first
online load caches everything needed for full offline operation (service
worker in `agrovision/offline/pwa/sw.js`). After that, diagnoses made with no
internet are journaled on-device and auto-sync when connectivity returns (no
manual sync button; status is always visible).

### What changes offline (honest answers)

1. **Image classifier** → quantized ONNX export (`mobilenetv2_quant.onnx`)
   running through `onnxruntime` (a Pyodide package), not TensorFlow.
2. **Explanation** → weights-based CAM (mathematically equivalent to Grad-CAM
   for MobileNetV2's GAP head), exported by `scripts/export_onnx.py`.
3. **Soil SHC OCR** → Tesseract/EasyOCR cannot run inside Pyodide, so offline
   soil input **degrades explicitly** to guided manual entry with a notice.
   Nothing silently drops — the module is honest about it.
4. **Weather** → network API, so offline weather is skipped and diagnosis
   proceeds on photo (+soil) alone.
5. **OpenCV** → every gatekeeper check has a **pure-NumPy fallback** that
   computes the *same* quantity (Laplacian variance, luma mean, green-pixel
   ratio, near-white ratio). `gatekeeper/backends.py` probes cv2 once and picks
   a backend; `tests/test_gatekeeper.py` asserts parity between the two.

---

## Training the classifier (you, on real data)

```bash
# Merge PlantVillage (38-class augmented) + PlantDoc + your India-specific
# rice/wheat/cotton/sugarcane data under one root:
#   <data>/train/<class>/...   <data>/val/<class>/...
python scripts/finetune.py --data ./merged_data --epochs 20 \
    --checkpoint assets/model/mobilenetv2_finetuned.h5
```

Then produce the offline artifacts:

```bash
python scripts/export_onnx.py --checkpoint assets/model/mobilenetv2_finetuned.h5
```

This writes `mobilenetv2_quant.onnx`, `mobilenetv2_cam.onnx`, and refreshes
`classes.json` (index order MUST match training order — the script preserves
the multilingual disease content already in the file).

---

## Tests

```bash
python -m pytest tests -q        # 36 tests, no network or GPU required
```

Highlights: NumPy-vs-OpenCV gatekeeper parity, OCR parser (incl. the `1:2.5`
pH ratio pitfall), i18n fallback guarantees, fusion normalization, sync-journal
behavior when the DB is down.

---

## Architectural notes (the non-obvious decisions)

- **Backend probe, not try/except per call.** `gatekeeper/backends.py` probes
  cv2 *once* at first use (including a 1×1 filter real-usage check, because some
  Pyodide wheels import but crash at runtime). Every check then calls the chosen
  backend — verdicts are reproducible online and offline.
- **Offline CAM ≠ placeholder.** Since MobileNetV2 ends in GAP→Dense, CAM =
  Grad-CAM up to a constant that vanishes on normalization. The ONNX export
  therefore carries real attribution, so the "explain yourself" layer is not
  silently weakened offline.
- **sklearn's role is confined to fusion.** The image classifier is TF/ONNX;
  scikit-learn is used only for the decision-layer scoring (a logistic
  calibrator trained on synthetic anchors from the hand-authored reference
  table), exactly as scoped.
- **OCR is label-matched, not free-form parsed.** `soil/parser.py` matches
  known SHC label strings (EN+HI) and pulls the trailing value — plus a
  plausibility gate — so a misread digit can't silently enter the pipeline.
- **No manual sync button.** `data/sync.py` journals offline records and flushes
  them whenever connectivity returns; failed flushes keep the journal intact.
- **Demo mode is explicit.** Without a checkpoint, `DemoBackend` runs the full
  UX deterministically and the UI shows a `model_warning` — the app never
  presents placeholder output as a real diagnosis.
- **Streamlit has no live-video loop**, so the gatekeeper is a capture →
  check → feedback → retake state machine on stills (as scoped).

## Disclaimer

Krishi Scan provides guidance only. It is not a substitute for a trained
agriculture officer — always confirm before applying any pesticide.
