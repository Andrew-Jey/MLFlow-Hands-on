# MLflow Tracking Quickstart

This project demonstrates the basic features of MLflow Tracking, including logging parameters, metrics, and models. It uses the classic Iris dataset to train a Logistic Regression model.

## Prerequisites

- Python 3.8 or later
- pip (Python package installer)

## Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd "MLFlow Hands-on/Quickstart"
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r .\Quickstart\requirements.txt
    ```

## Usage

### 1. Start the MLflow Tracking Server

Before running the training script, start the MLflow UI to view your experiments:

```bash
mlflow ui --port 5000
```

Open your browser and navigate to `http://localhost:5000`.

### 2. Run the Training Script

In a separate terminal window (ensure your virtual environment is activated), run the `quickstart.py` script:

```bash
python quickstart.py
```

This script will:
- Load the Iris dataset.
- Train a Logistic Regression model.
- Log hyperparameters, accuracy, and the trained model to MLflow.
- Register the model as "tracking-quickstart".

### 3. View Results

Go back to the MLflow UI at `http://localhost:5000`. You should see a new experiment named "MLflow Quickstart" with a new run containing your logged data.

## Project Structure

- `quickstart.py`: Main Python script for training and logging.
- `first_experiment.ipynb`: Jupyter Notebook version of the quickstart (for interactive exploration).
- `requirements.txt`: List of Python dependencies.
- `mlruns/`: Directory where MLflow stores tracking data (if using local file storage).
