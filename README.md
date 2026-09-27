# Smart Attendance System

Face recognition attendance built on the **existing trained classical ML model**.
No new model was introduced: the shipped `LogisticRegression` + `StandardScaler` +
`FeatureSelector` artifacts in `models/` are used exactly as they were trained.

## What it does

- **Live camera recognition** in the browser (`navigator.mediaDevices.getUserMedia`),
  with Start / Stop / Capture / Recognise controls
- **Automatic attendance** when a registered face is recognised, with one row per
  student per day enforced by a database constraint
- **Unknown-face rejection** so strangers never produce attendance
- **Dataset management** in the UI: add one student or import a ZIP of many
- **Dashboard, attendance history, reports and CSV export**
- **Prometheus metrics** at `/metrics`

## How recognition works

```
browser frame
  -> face detection (Haar cascade; whole frame is used when no face is found,
     which is exactly what train.py did)
  -> preprocessing (64x64 grayscale, histogram equalisation, Gaussian blur)
  -> feature extraction (HOG + LBP + geometric + statistical + shape) = 1805 dims
  -> StandardScaler
  -> FeatureSelector (1805 -> 715)
  -> LogisticRegression -> student id
  -> prototype verification (unknown-face rejection)
  -> attendance row
```

### Why there is a verification step

The trained classifier's `predict_proba` is saturated: it returns ≈1.0 for *any*
input, including pure noise, so its confidence cannot be used to decide whether
someone is a registered student. Identity is therefore confirmed by cosine
similarity against per-class prototypes built from the registered dataset in the
same scaled 1805-dim feature space. Both numbers are reported in every response
(`confidence` and `similarity`) together with the threshold, and the response also
carries `model_student_id` and `model_agreed` so the classifier's own answer is
never hidden.

Measured on the shipped dataset by `scripts/measure_inference.py`
(150 registered images, 48 noise/blank samples):

| Metric | Value |
| --- | --- |
| Classifier top-1 accuracy | 58.7% |
| Verified identity accuracy | 96.7% |
| Registered faces accepted | 97.3% |
| Unknown faces wrongly accepted | 0 / 48 |
| Registered vs unknown separation (AUC) | 1.000 |
| Recognition threshold | 0.08 |

Reproduce with:

```bash
python3 scripts/measure_inference.py    # writes models/verification_metrics.json
```

### Read these numbers honestly

**The table above is in-sample.** It is measured on `data/raw`, which is the same
150 images the shipped model was trained on, and the prototype cache is built
from those same images. It therefore answers "does the pipeline recognise the
images it was built from?", not "does it recognise a new photograph?". Do not
quote 96.7% as a generalisation result.

A real 60/20/20 stratified run of the training pipeline on the same data
(`python3 train.py --output-dir <dir>`, test split never used for model
selection) gives:

| Split | Images | Classifier accuracy | Weighted F1 |
| --- | --- | --- | --- |
| Train | 90 | — | — |
| Validation | 30 | 33.3% | 33.3% |
| **Test (held out)** | **30** | **13.3%** | **13.3%** |

The honest summary: **this feature pipeline does not generalise well to unseen
faces with the current 150-image dataset.** 5 students x 30 tightly-cropped
128x128 images is very little data for a 1805-dimensional handcrafted feature
space, and the held-out classifier is close to the 20% chance baseline for
5 classes.

What actually works better in practice, and what the verification step buys you,
is **re-registration**: every image added through the Dataset page extends that
student's prototype set, so a student is recognised reliably against the exact
images captured for them. The `similarity` score and threshold are what govern
attendance, and unknown faces are rejected (`0 / 48` above) because their
similarity does not reach the threshold - not because the classifier is accurate.

To improve generalisation, in rough order of impact:

1. Capture more images per student, ideally 20+ at higher resolution, from
   varied angles and lighting (`python3 train.py --output-dir <dir>` retrains).
2. Re-capture so faces are actually **detectable** (see the face-detection note
   below - the current dataset has none, so training uses whole frames).
3. Raise image resolution: 64x64 discards most facial detail.

**The threshold is dataset-specific.** `0.08` was measured against this dataset.
If you register a different set of faces, re-run `scripts/measure_inference.py`
and adjust `RECOGNITION_THRESHOLD` to suit. With prototypes that carry little
information (for example the generated placeholder frames the test suite falls
back to when `data/raw` is empty) a fixed threshold cannot separate a real face
from noise, and unknown faces will be accepted. The test suite reflects this: the
unknown-rejection tests only run against a real dataset.

Because the classifier is weak on unseen data, **the recorded identity comes from
the verification step**; the model output is still reported alongside it.

## Requirements

- Python 3.9+ (3.11 recommended)
- Node.js 18+ and npm (only to rebuild the frontend)

The camera requires a **secure context**: `https://` in production, or
`http://localhost` during development. `localhost` counts as secure, so the demo
works on a laptop without certificates.

## Run it

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd frontend && npm install && npm run build && cd ..
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd frontend; npm install; npm run build; cd ..
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. The backend serves the compiled frontend and the API
together, so one origin covers everything.

### Frontend dev server (hot reload)

```bash
cd frontend && npm run dev     # http://localhost:5173, proxies /api to :8000
```

### Docker

```bash
docker compose up --build      # http://localhost:8000
```

The image builds the frontend in a Node stage and serves it from FastAPI, so there
is a single origin and a single URL.

## Tests

```bash
python -m pytest tests/ -q     # backend: 127 passed
cd frontend && npm test        # frontend: 19 passed
ruff check .                   # critical lint rules
cd frontend && npm run build
```

Tests run against a temporary copy of the dataset, a temporary SQLite database and a
temporary prototype cache, so the real `data/` directory and `models/prototypes.npz`
are never touched. They cover the database (including duplicate prevention), the
dataset manager, preprocessing and feature finiteness, the model stack, the verifier's
unknown-face rejection, the training split and its leakage guards, and the full API
including the camera upload path. No test needs a real student image or a private
credential: a small non-biometric dataset is synthesised when `data/raw` is empty, and
the tests that genuinely require real faces skip themselves with a reason.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

- **backend** - install, build the frontend, `ruff check .`, byte-compile, then
  explicitly validate the preprocessing pipeline, the 1805-feature extractor, the
  model artifacts (selector output width vs `model.n_features_in_`), the API import,
  the training entry point, and the full pytest suite. Fails if any image, dataset
  folder or database is tracked in git.
- **frontend** - `npm ci`, `npm test`, `npm run build`, then fails if the built
  bundle contains a hardcoded `localhost:8000`.
- **docker** - builds `docker/Dockerfile` with Buildx on a clean runner, validates
  the compose file, boots the container and curls `/health`, `/api/health`, the SPA
  and `/metrics`.

`.github/workflows/cd.yml` publishes the validated image to GitHub Container
Registry on a `v*` tag (or manually). It refuses to publish unless CI passed for
that exact commit, authenticates with the automatic `GITHUB_TOKEN`, and requests
only `contents: read` + `packages: write`. No registry password, API key or
deployment secret is stored in the repository.

## API

All endpoints live under `/api`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | model, database, verifier and dataset status |
| GET | `/api/model/info` | model metadata and verification method |
| POST | `/api/predict` | multipart image upload |
| POST | `/api/predict-base64` | JSON `{ "image_data": "data:image/jpeg;base64,…" }` |
| GET/POST | `/api/students` | list / create students |
| GET/PATCH/DELETE | `/api/students/{id}` | read, update, delete (`?remove_images=true` also deletes images) |
| POST | `/api/students/upload` | add images for one student |
| POST | `/api/students/upload-zip` | import `<STUDENT_ID>/<image>` ZIPs |
| GET | `/api/dataset/stats` | real dataset statistics (`?deep=true` decodes every image) |
| POST | `/api/dataset/rebuild-verifier` | rebuild prototypes after a dataset change |
| GET/POST | `/api/attendance` | history (filter by date, student, status) / manual mark |
| GET | `/api/attendance/today` | today's attendance |
| GET | `/api/attendance/export.csv` | CSV export |
| GET | `/api/dashboard/stats` | dashboard counters |
| GET | `/api/stats/predictions` | recognition statistics |
| GET | `/metrics` | Prometheus metrics |
| GET | `/health`, `/dashboard/stats` | root aliases of the two non-conflicting read-only endpoints |
| GET | `/docs` | interactive API documentation |

## Registering students

The model can only recognise the identity labels it was trained on
(`S0001`–`S0005` here). Registering a new student adds their images to the dataset
and the verification prototypes, but recognising them reliably requires retraining
the classifier with the new labels.

Dataset layout:

```
data/raw/
  S0001/  image1.jpg  image2.jpg ...
  S0002/  ...
```

A dataset ZIP must use that layout. `__MACOSX/`, `.DS_Store` and `._*` entries are
ignored automatically, and names that would escape the dataset directory are refused.

## Privacy

Face images are biometric personal data.

- `data/raw/` and any `dataset/` folder are **git-ignored** and must stay that way.
  CI fails the build if any image or database file is tracked.
- Nothing is baked into the Docker image; images are uploaded at runtime.
- Captured camera frames are sent to the backend for recognition and are **not**
  written to disk.
- Recognition logs store the student id, decision, score and latency, never the image.

Before deploying with real faces, make sure the storage behind it is access
controlled, and get consent from the people in the dataset.

## Dependency pinning

`requirements.txt` pins `scikit-learn==1.7.2`, the version the artifacts in
`models/` were pickled with. Unpickling a `StandardScaler` or `LogisticRegression`
across scikit-learn versions can fail outright and leave the app running with no
model, so do not loosen that pin without re-running
`scripts/measure_inference.py` and re-checking the numbers above.

## Configuration

Copy `.env.example` to `.env`. Paths are resolved relative to the project root, so
the same file works on macOS, Windows and Linux. The main values:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATASET_PATH` | `./data/raw` | folder of per-student image folders |
| `DATABASE_URL` | sqlite in `data/database` | SQLite or PostgreSQL |
| `MODEL_PATH`, `SCALER_PATH`, `FEATURE_SELECTOR_PATH`, `LABEL_ENCODER_PATH` | `models/*` | existing trained artifacts |
| `RECOGNITION_THRESHOLD` | `0.08` | similarity needed to record attendance |
| `FACE_POLICY` | `whole_frame` | `whole_frame` or `require_face` (see below) |
| `ENVIRONMENT` | `development` | `production` restricts CORS to the listed origins |
| `API_BASE_URL` | *(empty)* | public API origin, for split frontend/backend hosting |
| `FRONTEND_URL`, `CORS_ORIGINS` | same origin | set when hosting the frontend separately |
| `FRONTEND_DIST` | `./frontend/dist` | compiled frontend to serve |

## Face detection, and a measured limitation

`FaceDetector` uses a Haar cascade resolved through
`src/preprocessing/cascade_locator.py`, which prefers the copies **vendored in
`src/preprocessing/cascades/`** and only then falls back to `cv2.data`. This
matters: `requirements.txt` used to pin both `opencv-python` and
`opencv-contrib-python`, and because both write into the same `cv2` directory a
fresh Linux install ended up with an incomplete `cv2/data`. `CascadeClassifier`
then returned an *empty* classifier, so face detection was silently disabled in
CI and in Docker while `/api/health` still reported `healthy`. `contrib` is now
removed (nothing in the project uses a contrib module) and the cascades are
vendored, so detection behaves identically on macOS, Windows, Linux and Docker.

`opencv-python` is additionally pinned to `<5.0.0`. OpenCV 5.0 **removed the
legacy Haar API** - `cv2.CascadeClassifier` no longer exists and the cascade
XMLs are not shipped. An unbounded `>=4.10.0` resolved to 5.0, which broke the
backend with `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`
and killed the container at startup, because `FaceDetector()` is built in the
FastAPI startup hook. `tests/test_cascades.py` pins the constraint and asserts
the API exists so a future bump cannot silently reintroduce the outage.
`tests/test_cascades.py` covers it, including a simulated broken `cv2/data`.

`haarcascade_mcs_nose` and `haarcascade_mcs_mouth` are **not** shipped by current
OpenCV wheels, so the nose and mouth sub-detectors are unavailable and their
sub-features fall back to zeros. That is deliberate: the shipped model was
trained without them, and adding them now would change the 1805-dimensional
feature vector and invalidate `models/`.

Detection parameters are deliberately left at the values the shipped model was
trained with, because loosening them changes the feature geometry and measurably
hurts accuracy.

**Measured on the shipped dataset, Haar detects a face in 0 of 150 images.**
Every training feature was therefore computed from a whole 128x128 frame rather
than a face crop. Two consequences, both handled explicitly:

- `FACE_POLICY=whole_frame` (default) classifies the whole frame when no face is
  found, which keeps inference in the same distribution as training. Every
  response reports `face_detected` and `faces_detected` so a client can tell.
- `FACE_POLICY=require_face` refuses such frames with
  `{"success": false, "status": "no_face_detected"}` and never marks attendance.
  Safer, but recall will collapse until the dataset is re-captured with
  detectable faces, because the model has never seen a crop.

When several faces are present the **largest box wins** (deterministic), and
`faces_detected` reports the count so the UI can ask the person to stand alone.

This also means live camera frames, which usually *do* contain a detectable face,
are preprocessed differently from the training images. That path could not be
measured here, because it needs real webcam captures - if the demo accuracy
looks wrong on camera but right on uploaded photos, set `FACE_POLICY` and
re-measure.

## Retraining

`train.py` is the training entry point. It is a full classical-ML pipeline:
preprocess -> extract 1805 features -> stratified 60/20/20 split -> fit scaler
and feature selector **on the training split only** -> tune each classical
model -> select on validation weighted F1 -> evaluate once on the held-out test
set -> save artifacts -> log real metrics to MLflow.

```bash
python3 train.py --output-dir models_v2   # never overwrites models/ by accident
```

`--output-dir` matters: the shipped `models/*.pkl` are the artifacts the API
loads, and the default `models` would overwrite them. Test selection never
considers `*_test` rows, and the scaler/selector are fitted on `X_train` only;
`tests/test_training.py` guards both.

## Testing

```bash
python3 -m pytest -q          # backend, 108 tests
cd frontend && npm test       # frontend, 19 tests
cd frontend && npm run build
ruff check .                  # critical lint rules
```

No test needs a real student image or a private credential. The suite
synthesises a small non-biometric dataset when `data/raw` is empty and skips the
claims that genuinely require real faces.


## Project layout

```
config/            YAML configuration (64x64 target size lives here)
src/api/           FastAPI app, schemas, Prometheus metrics
src/core/          settings and path resolution
src/data/          database and dataset manager
src/features/      HOG, LBP, statistical, shape feature extractors
src/models/        predictor wrapping the existing artifacts
src/preprocessing/ image processor, face detector, vendored Haar cascades
src/verification/  prototype verifier for unknown-face rejection
frontend/          React + Vite single page app
models/            the existing trained artifacts
scripts/           measure_inference.py
tests/             pytest suite
```

`preprocessing.target_size` in `config/config.yaml` is the single source of truth
for the model input size. Constructing `ImageProcessor` with its own defaults would
silently use 128x128 and destroy accuracy, so call sites use the configured value.

## Notes and known limitations

- `dlib` is optional and not installed; face detection uses the bundled OpenCV Haar
  cascade, which is what training used. OpenCV logs a warning for two optional
  cascades it does not ship; they are not used.
- Because the training images are 128x128, Haar often finds no face and the whole
  frame is used, matching training behaviour.
- The classifier itself is weak (58.7%); the verification step is what makes
  attendance trustworthy. Retraining the classifier on a larger, better dataset is
  the real improvement path.
