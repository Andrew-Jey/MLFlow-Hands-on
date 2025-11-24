# Deep Learning with MLflow and PyTorch

This project demonstrates how to train a PyTorch Neural Network on the FashionMNIST dataset and track the experiments using [MLflow](https://mlflow.org/).

It showcases:
- Automatic logging of system metrics (CPU, GPU, Memory).
- Logging of training parameters and hyperparameters.
- Tracking of training and validation metrics (Loss, Accuracy) across epochs.
- Model checkpointing and final model logging.

## Project Structure

- `deeplearning.py`: The main script that trains the Neural Network.
- `requirements.txt`: Pip requirements file.

## Prerequisites

- Python 3.8+
- MLflow
- PyTorch
- Torchvision
- Pandas
- Psutil (for system metrics)

## Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd "MLFlow Hands-on/Deep Learning"
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

The project uses a local SQLite database for tracking (`mlflow.db`). Before running the training script, start the MLflow UI to view your experiments:

```bash
mlflow ui --port 5000
```

Open your browser and navigate to `http://localhost:5000`.

### 2. Run the Training Script

In a separate terminal window (ensure your virtual environment is activated), run the `deeplearning.py` script:

```bash
python deeplearning.py
```

This script will:
- Download the FashionMNIST dataset (if not present).
- Create an MLflow experiment named "Deep Learning Experiment".
- Train a Neural Network for 5 epochs.
- Log system metrics, loss, accuracy, and model checkpoints to MLflow.

### 3. View Results

Go back to the MLflow UI at `http://localhost:5000`. You should see a new experiment named "Deep Learning Experiment" with runs containing your logged data.
