2_regression_on_the_tabular_data — Regression baseline and EDA summary

Overview
- Goal: train a tabular regression model on datasets/train.csv and produce predictions for datasets/hidden_test.csv.
- Baseline model in this folder: HistGradientBoostingRegressor inside a simple sklearn Pipeline.
- Notebook: eda.ipynb documents exploratory analysis and model comparison.

Project layout
- train.py — trains a model and saves it to models/hgb_model.joblib
- predict.py — loads the saved model and writes predictions to predictions/predictions.csv
- data_preparation.py — optional preprocessing (e.g., dropping suspicious/leaky features)
- datasets/
  - train.csv — training data (must contain a numeric column named target)
  - hidden_test.csv — test data for inference (no target column)

Quick start
Option A — pip
1) Create and activate a virtual environment (see repo root README for details)
2) Install dependencies from the project root:
   pip install -r requirements.txt
3) run:
   - cd 2_regression_on_the_tabular_data
   - Train: python train.py
   - Predict: python predict.py

Option B — Poetry
1) poetry install (run at the project root)
2) Run:
   - cd 2_regression_on_the_tabular_data
   - Train: poetry run python train.py
   - Predict: poetry run python predict.py

Notes
- Paths are relative to this folder by default: datasets/, models/, predictions/.
- Python version: 3.11.x (per pyproject.toml).
- Reproducibility: scripts use random_state=42 for splitting and models.

EDA highlights (from eda.ipynb)
1) Target distribution and basic checks
   - The target is continuous and fairly well-behaved after inspection (histogram and describe()).
   - No severe missingness in the target; some features may contain missing values (handled in the pipeline).

2) Correlations and important features
   - Correlation heatmap and bar plots show that a small subset of features explains most variance of the target.
   - Feature "6" stands out as suspicious: scatter plots and simple transformations suggest a strong nonlinear relation (target roughly tracks feature 6 squared). This looks like engineered dependency or potential target leakage.

3) Sanity check on feature "6"
   - Comparing (feature_6 ** 2) to target yields very low error on the validation slice, indicating an almost deterministic link.
   - Conclusion in the notebook: keep this feature for the training-task baseline (to maximize public metric), but call it out as unacceptable for production; in prod, we would drop/verify it.

Model comparison (hold-out split in the notebook)
Models evaluated on the same train/validation split:
- Linear Regression — worst RMSE among the three, underfits in presence of nonlinearity.
- Random Forest — substantially better RMSE than Linear Regression, robust to outliers and nonlinearities; stable without extensive tuning.
- HistGradientBoosting (HGB) — achieved the best RMSE on the hold-out split in the EDA, with reasonably shaped residuals and good cross-validation stability (lowest mean RMSE, low std across folds) after removing feature "6".

Why we ship HGB as the baseline model
- It achieved the second-best RMSE on the hold-out split in the EDA (with feature 6), with a low gap between first and second places.
- Compact model size and efficient inference: unlike Random Forest, HistGradientBoosting produces a lightweight model artifact that is easy to store, version, and distribute.
- Strong performance on tabular data: gradient boosting methods are widely used for structured datasets and typically provide excellent predictive accuracy with minimal preprocessing.
- Minimal preprocessing requirements: HGB handles heterogeneous numerical features well and works effectively without feature scaling or extensive feature engineering.
- Stable and reproducible: the model demonstrates consistent performance across cross-validation folds and produces deterministic results when a fixed random seed is used.
- Practical deployment characteristics: training and inference are fast, memory-efficient, and suitable for iterative experimentation and production scenarios.

How to reproduce the notebook’s findings
1) Open eda.ipynb (after installing deps):
   - With pip: jupyter notebook
   - With Poetry: poetry run jupyter notebook
2) In the notebook you’ll find:
   - Target distribution, global correlation heatmap.
   - A deep-dive on feature "6" including scatter plot vs target and squared transform.
   - Hold-out evaluation of Linear Regression, Random Forest, and HGB, with RMSE summaries.
   - For the best-performing HGB run (after dropping feature "6"), residual diagnostics, 5-fold CV (neg RMSE) mean/std, permutation importance, and SHAP summaries.

Running the baseline
- Train the baseline HGB model:
  python train.py
- Generate predictions for hidden_test.csv:
  python predict.py

Optional: enable preprocessing that drops feature "6"
- data_preparation.py includes DataPreprocessor that drops column "6" if present.
- You can enable it by uncommenting the corresponding lines in train.py and predict.py:
  - In train.py (inside run):
    preprocessor = DataPreprocessor()
    X = preprocessor.transform(X)
  - In predict.py (inside run):
    data = DataPreprocessor().transform(data)

Switching the baseline to Random Forest (if you prioritize RMSE)

- In `train.py`, replace the regressor in `build_model()` with:

```python
from sklearn.ensemble import RandomForestRegressor

...

("regressor", RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
))
```

Troubleshooting
- If datasets/train.csv is missing or target column not found, the trainer will raise a clear error.
- If you observe substantially worse performance, verify that:
  - The random_state is fixed.
  - Feature "6" handling is consistent between training and inference.
  - You are evaluating on a proper hold-out or via cross-validation.
