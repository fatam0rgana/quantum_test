Quantum Test — How to run with pip or Poetry

Overview
- Python requirement: 3.11.x (as specified in pyproject.toml: >=3.11,<4.0, I used 3.11.6 locally)
- This repository contains three tasks:
  1. 1_counting_islands — pure Python solution with tests.
  2. 2_regression_on_the_tabular_data — training and prediction scripts for a tabular regression model.
  3. 3_MNIST_classifier — simple (mocked) model adapters for a digit classifier.

Install dependencies

Option A — pip (virtual environment recommended)
1) Create and activate a virtual environment (examples for common shells):
   - On Linux/macOS (bash/zsh):
     python3 -m venv .venv
     source .venv/bin/activate
   - On Windows (PowerShell):
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1

2) Install project requirements from the generated requirements.txt:
   pip install -r requirements.txt

Option B — Poetry
1) Ensure Poetry is installed:
   - pipx install poetry
   - or follow: https://python-poetry.org/docs/#installation

2) Install dependencies (project uses package-mode = false):
   poetry install

3) Run any command via Poetry using:
   poetry run <command>

How to run the tasks

1) Counting islands (1_counting_islands)
   - cd 1_counting_islands
   - Run tests with pip environment:
     python -m unittest discover -s tests -p "tests_*.py"
   - Run tests with Poetry:
     poetry run python -m unittest discover -s tests -p "tests_*.py"

2) Regression on tabular data (2_regression_on_the_tabular_data)
   Directory contains:
   - train.py — trains a RandomForestRegressor and saves to models/rf_model.joblib
   - predict.py — loads the saved model and writes predictions to predictions/predictions.csv
   - datasets/ — expected to contain train.csv and hidden_test.csv (hidden_test.csv is already present)

   With pip:
   - cd 2_regression_on_the_tabular_data
   - Train:
     python train.py
   - Predict (after training):
     python predict.py

   With Poetry:
   - cd 2_regression_on_the_tabular_data
   - Train:
     poetry run python train.py
   - Predict (after training):
     poetry run python predict.py

   Notes:
   - If you run from Linux/macOS, replace backslashes with forward slashes in paths.
   - The scripts use default relative paths (datasets/, models/, predictions/). You can modify paths by passing constructor args inside the scripts if needed.

3) MNIST classifier stubs (3_MNIST_classifier)
   The models are mocked; they validate input shapes and return deterministic defaults when the real model is not loaded.
   - No training script is provided; you can import and use classes from 3_MNIST_classifier/digit_classifier/models.py.
   - Example (pip):
     python -c "from digit_classifier.models import RandomModel; import numpy as np; print(RandomModel().predict(np.zeros((28,28,1))))"
   - Example (Poetry):
     poetry run python -c "from digit_classifier.models import RandomModel; import numpy as np; print(RandomModel().predict(np.zeros((28,28,1))))"

Working with the EDA notebook
The 2_regression_on_the_tabular_data/eda.ipynb notebook can be opened after installing dependencies:
   - With pip: jupyter notebook
   - With Poetry: poetry run jupyter notebook
Then open the notebook file in the browser UI.

Troubleshooting
- If you see ModuleNotFoundError when running from a subdirectory, run the command from the project root so that relative imports and paths resolve correctly.
- Ensure your active interpreter matches Python 3.11.
- If Poetry complains about Python version, specify the correct interpreter:
  poetry env use python3.11
- If you prefer conda, create an env with Python 3.11 and then use pip install -r requirements.txt within it.
