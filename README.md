# Bank Marketing Response Prediction

A learning project for estimating whether a client will subscribe to a term deposit. The model prediction point is **before the current marketing call begins**. This repository contains exploratory analysis, chronological model comparison, a reusable Scikit-learn pipeline, a saved experimental artifact, and a FastAPI prediction endpoint.

## Project status

The API and training workflow are implemented and tested. The current model is **not ready for operational outreach**: no validation candidate met the requested precision floor of 0.50, the chronological segments have a large subscriber-rate shift, and the last segment was inspected and scored in earlier notebook work. Its current results are therefore post-exposure diagnostics, not an untouched final estimate.

## Dataset and prediction setup

- Dataset: UCI Bank Marketing `bank-additional-full.csv` (41,188 rows; 20 predictor columns and target `y`).
- Target: `y`, with labels `yes` and `no`.
- Excluded predictors: `duration` (known after the call) and `campaign` (includes current campaign contact under the project's strict pre-call policy).
- Preprocessing: median imputation and standard scaling for numeric inputs; most-frequent imputation and one-hot encoding for categorical inputs. The preprocessing is inside each pipeline and is fitted on training rows only.
- Data order is preserved; no random shuffling is used.

The split is 64% model training, 16% validation and 20% final segment:

| Segment | Rows | Subscriber rate |
|---|---:|---:|
| Training | 26,360 | 4.79% |
| Validation | 6,590 | 12.72% |
| Final segment | 8,238 | 30.83% |

The last rate is included because it was already inspected in prior notebook runs. The final segment can no longer be considered untouched. UCI describes `bank-additional-full.csv` as ordered by date from May 2008 to November 2010; the file does not contain an exact timestamp field, so this project preserves that supplied order. See the [UCI dataset documentation](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing).

## Validation model comparison

All entries below use the validation segment only. Precision, recall and F1 use the default 0.50 threshold. Average Precision (AP) is the primary ranking measure because of class imbalance; ROC-AUC is also reported.

| Candidate | ROC-AUC | AP | Precision | Recall | F1 | Brier | Mean predicted probability |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest, balanced | 0.5926 | **0.1572** | 0.1700 | 0.3341 | 0.2254 | 0.2277 | 0.4699 |
| Random Forest, unweighted | **0.6000** | 0.1568 | 0.0000 | 0.0000 | 0.0000 | 0.1161 | 0.0523 |
| Random Forest, balanced, larger leaf | 0.5877 | 0.1550 | 0.1669 | 0.3783 | 0.2316 | 0.2331 | 0.4775 |
| Random Forest, balanced, shallower | 0.5844 | 0.1547 | 0.1615 | 0.3604 | 0.2230 | 0.2307 | 0.4739 |
| Decision Tree | 0.5662 | 0.1441 | 0.1467 | 0.8663 | 0.2510 | 0.2681 | 0.5162 |
| Logistic Regression | 0.5731 | 0.1358 | 0.0000 | 0.0000 | 0.0000 | 0.1266 | 0.0067 |
| Class-weighted Logistic Regression | 0.5729 | 0.1354 | 0.0270 | 0.0024 | 0.0044 | 0.1510 | 0.0836 |
| Chronologically calibrated Random Forest | 0.4781 | 0.1185 | 0.0000 | 0.0000 | 0.0000 | 0.1170 | 0.0500 |

The balanced Random Forest retains the best AP among these candidates, but its margin over the unweighted forest is small. Its mean validation probability (46.99%) is far above the observed validation rate (12.72%). Removing class weights lowers the mean probability (5.23%) and improves Brier score, but at the default threshold it predicts no subscribers. The time-series sigmoid calibration experiment did not improve ranking or Brier score. Calibration and ranking remain poor under the observed time shift.

### Threshold objective

The requested threshold objective is **maximize recall subject to validation precision of at least 0.50**. No tested candidate reaches 0.50 precision while predicting any subscribers. The highest observed precision is 0.1826 (unweighted Random Forest, recall 0.3628, threshold approximately 0.057). Therefore no threshold can satisfy the stated outreach constraint with the current candidates.

The highest-AP candidate is carried forward only for a reproducible diagnostic artifact. Its API classification threshold is set to 0.50 as a provisional default, not as a threshold selected to meet the outreach constraint. Do not use it for production outreach without resolving this model-readiness issue.

## Final-segment diagnostic (previously exposed)

The highest-AP validation candidate was fixed before this diagnostic. The model was refit on training plus validation, then evaluated once by the current clean notebook run on the last segment. The segment had already been repeatedly inspected in earlier notebook versions, so these are **post-exposure results**.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.6959 |
| Average Precision | 0.4840 |
| Precision at 0.50 | 0.3514 |
| Recall at 0.50 | 0.9268 |
| F1 at 0.50 | 0.5096 |
| Accuracy | 0.4500 |
| Brier score | 0.2642 |
| Actual subscriber rate | 30.83% |
| Predicted subscriber rate | 81.32% |

Confusion matrix (rows: actual `no, yes`; columns: predicted `no, yes`):

```text
[[1353, 4345],
 [ 186, 2354]]
```

The final segment's later-period distribution differs sharply from training and validation. Recall rises, but the 0.50 threshold overpredicts subscribers and precision remains below the requested floor. The post-exposure test scores must not be compared as if they were an unbiased estimate of future deployment performance.

## Interpretation

For the carried-forward Random Forest, the largest impurity-based importances in this fit are `euribor3m`, `cons.conf.idx`, `cons.price.idx`, `emp.var.rate` and `nr.employed`, followed by month-related indicators. These correlated economic variables describe how the fitted trees split this dataset; importance is not causal evidence or a business effect. See `reports/feature_importance.csv`. SHAP was not added because the current model comparison does not justify extra interpretability tooling.

## Structure

```text
bank-marketing-ml/
|-- data/raw/                  # Existing local UCI files
|-- notebooks/
|   `-- 01_data_inspection.ipynb
|-- src/bank_marketing/
|   |-- preprocessing.py       # Loading, exclusions, chronological split, transforms
|   |-- models.py              # Candidate pipelines
|   |-- train.py               # Train/validation comparison only
|   |-- evaluate.py            # Ranking, threshold and calibration metrics
|   |-- finalize.py            # Refit and one final-segment diagnostic
|   |-- predict.py             # Input validation and one-row inference
|   `-- api.py                 # FastAPI health and prediction routes
|-- models/                    # Locally generated joblib artifact
|-- reports/                   # CSV/JSON results and feature importance
|-- frontend/                  # React + Vite interactive dashboard
|-- tests/                     # Unit and API tests
|-- requirements.txt
|-- pyproject.toml
```

## Setup and commands

The raw UCI dataset is excluded from this repository. Download the Bank Marketing dataset from the [UCI dataset page](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing), extract `bank-additional-full.csv`, and place it at `data/raw/bank-additional-full.csv`. Keep the downloaded archive and extracted data local; the project does not automatically fetch or commit the dataset.

From PowerShell in the project directory, create a Python 3.13 environment and install the project:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
pytest -q
```

The included model artifact was created with scikit-learn 1.9.1, which is pinned in `requirements.txt` for compatibility. The other Python dependency entries specify minimum versions rather than a complete lockfile.

Compare models on training and validation only (does not read or evaluate the final segment):

```powershell
python -m bank_marketing.train
```

The notebook was executed through Jupyter `nbconvert` from a clean kernel. Do not run the notebook file as a normal Python script. The final-evaluation command below refits a named candidate on training plus validation and evaluates the last segment; it is a diagnostic for this previously exposed dataset, not a fresh test protocol:

```powershell
python -m bank_marketing.finalize --model random_forest_balanced --threshold 0.5
```

The selected 3.2 MiB joblib artifact is included at `models/selected_model.joblib`; the final metrics JSON is `reports/final_test_metrics.json`. The API can load the included artifact without retraining. Regenerating it with the command above refits and reevaluates the previously exposed final segment. Joblib artifacts use Python pickle internally. Load only artifacts produced by this project or another trusted source.

The frontend metrics payload is generated from the local dataset and saved reports:

```powershell
python -m bank_marketing.dashboard_metrics
```

This exports `reports/dashboard_metrics.json`; it does not use the final segment for model selection. The API serves it from `GET /dashboard/metrics`.

## FastAPI

Start the local API from the project directory after generating the artifact:

```powershell
uvicorn bank_marketing.api:app --reload
```

- `GET /healthz` reports whether the model artifact loaded.
- `GET /dashboard/metrics` serves verified dataset, validation, calibration and post-exposure diagnostic values.
- `POST /predict` validates the full 18-feature schema and returns a class and probability.

By default, CORS allows `http://localhost:5173` and `http://127.0.0.1:5173`. Set `BANK_MARKETING_CORS_ORIGINS` to a comma-separated list to configure other local origins. `BANK_MARKETING_MODEL_PATH` and `BANK_MARKETING_METRICS_PATH` can override the local artifact and metrics file paths.

Example (all feature keys are required; `duration`, `campaign` and `y` must not be sent):

```json
{
  "features": {
    "age": 56, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon", "pdays": 999,
    "previous": 0, "poutcome": "nonexistent",
    "emp.var.rate": 1.1, "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4, "euribor3m": 4.857,
    "nr.employed": 5191.0
  }
}
```

A prediction response contains `predicted_class`, `subscription_probability`, and the artifact threshold. The threshold is provisional and does not meet the requested validation precision floor.

## React dashboard

Install and run the frontend from a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Use Node.js with npm; `package-lock.json` records the frontend dependency versions, and `npm ci` installs from that lockfile.

The Vite page uses `http://localhost:8000` by default. Copy `frontend/.env.example` to `frontend/.env.local` and set `VITE_API_BASE_URL` if the API runs elsewhere. The frontend requests dashboard evidence from `/dashboard/metrics` and sends every prediction to `/predict`; it does not contain a model copy or fabricate results.

Build and browser smoke-test (with FastAPI and Vite running; Chrome is required):

```powershell
npm run build
npm run test:smoke
```

The smoke test uses the locally installed Chrome. Set the `CHROME_EXECUTABLE_PATH` environment variable if Chrome is installed at a different path. It checks the overview, real API prediction, API-unavailable state, model performance view, mobile navigation and horizontal overflow.

## Tests and remaining work

The test suite covers chronological partition boundaries, feature exclusions, metrics, feature schema validation, and API health/prediction behavior. Current known limitations: the validation objective is infeasible for these candidates; the source has no explicit timestamp column; temporal subscriber prevalence changes sharply; the final segment was already exposed; and the API artifact is a student-project diagnostic, not an approved outreach model. A genuinely untouched evaluation requires a new future or independent dataset and a model that meets the agreed precision constraint on validation first.
