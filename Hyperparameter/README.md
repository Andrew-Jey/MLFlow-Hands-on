# Hyperparameter Tuning with MLflow and Optuna

This project demonstrates how to perform hyperparameter tuning using [Optuna](https://optuna.org/) and track the experiments using [MLflow](https://mlflow.org/).

The code is based on the [MLflow Hyperparameter Tuning Tutorial](https://mlflow.org/docs/latest/ml/getting-started/hyperparameter-tuning/).

## Prerequisites

- Python 3.8+
- MLflow
- Optuna
- Scikit-learn
- Pandas

## Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd "MLFlow Hands-on/Hyperparameter"
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Start the MLflow Tracking Server

Before running the training script, start the MLflow UI to view your experiments:

```bash
mlflow ui --port 5000
```

Open your browser and navigate to `http://localhost:5000`.


### 2. Run the Training Script

In a separate terminal window (ensure your virtual environment is activated), run the `hyperparameter.py` script:

```bash
python hyperparameter.py
```

This script will:
- Load the California Housing dataset.
- Create an MLflow experiment named "Hyperparameter Tuning Experiment".
- Run an Optuna study to optimize hyperparameters for a Random Forest Regressor.
- Log parameters, metrics, and models to MLflow.
- Register the best performing model.

### 3. View Results

Go back to the MLflow UI at `http://localhost:5000`. You should see a new experiment named "Hyperparameter Tuning Experiment" with a new run containing your logged data.

## Project Structure

- `hyperparameter.py`: The main script that performs hyperparameter tuning on a Random Forest Regressor using the California Housing dataset.
- `requirements.txt`: Pip requirements file.