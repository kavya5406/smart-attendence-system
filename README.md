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

**The threshold is dataset-specific.** `0.08` was measured against this dataset.
If you register a different set of faces, re-run `scripts/measure_inference.py`
and adjust `RECOGNITION_THRESHOLD` to suit. With prototypes that carry little
information (for example the generated placeholder frames the test suite falls
back to when `data/raw` is empty) a fixed threshold cannot separate a real face
from noise, and unknown faces will be accepted. The test suite reflects this: the
unknown-rejection tests only run against a real dataset.

These are measurements of the existing artifacts, not target or invented numbers.
Because the classifier is only 58.7% accurate on its own, **the recorded identity
comes from the verification step**; the model output is still reported alongside it.

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
python -m pytest tests/ -q
```

Tests run against a temporary copy of the dataset, a temporary SQLite database and a
temporary prototype cache, so the real `data/` directory and `models/prototypes.npz`
are never touched. They cover the database (including duplicate prevention), the
dataset manager, preprocessing and feature finiteness, the model stack, the verifier's
unknown-face rejection, and the full API including the camera upload path.

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
| `FRONTEND_URL`, `CORS_ORIGINS` | same origin | set when hosting the frontend separately |
| `FRONTEND_DIST` | `./frontend/dist` | compiled frontend to serve |

## Project layout

```
config/            YAML configuration (64x64 target size lives here)
src/api/           FastAPI app, schemas, Prometheus metrics
src/core/          settings and path resolution
src/data/          database and dataset manager
src/features/      HOG, LBP, statistical, shape feature extractors
src/models/        predictor wrapping the existing artifacts
src/preprocessing/ image processor and face detector
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
