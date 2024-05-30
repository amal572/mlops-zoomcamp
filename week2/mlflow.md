## Week 2: Introduction to MLflow, ML experiments and model registry

## Introduction:

### Definitions:
1. ML experiment: the process of building an ML model; The whole process in which a Data Scientist creates and optimizes a model
2. Experiment run: each trial in an ML experiment; Each run is within an ML experiment
3. Run artifact: any file associated with an ML run: Examples include the model itself, package versions...etc; Each Artifact is tied to an Experiment
4. Experiment metadata: metadata tied to each experiment

### Experiment tracking:
Keeping track of all the relevant information from an ML experiment; varies from experiment to experiment. Experiment tracking helps with Reproducibility, Organization and Optimization

Tracking experiments in spreadsheets helps but falls short in all the key points.

### MLflow:
It's a Python package with four main modules:

* Tracking
* Models
* Model registry
* Projects (Out of scope of the course)

### Tracking experiments with MLflow:
MLflow organizes experiments into runs and keeps track of any variables that may affect the model as well as its result; Such as: Parameters, Metrics, Metadata, the Model itself...

MLflow also automatically logs extra information about each run such as: Source Code, Git Commit, Start and End time and Author.

### Installing MLflow:
pip: ```bash
   pip install mlflow
```
conda:```bash
   conda install -c conda-forge mlflow
```

### Interacting with MLflow:
MLflow has different interfaces, each with their pros and cons. We introduce the core functionalities of MLflow through the UI.

### MLflow UI:
To run the MLflow UI locally we use the command:

```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
```
The backend storage is essential to access the features of MLflow, in this command we use a SQLite backend with the file mlflow.db in the current running repository. This URI is also given later to the MLflow Python API mlflow.set_tracking_uri.

By accessing the provided local url we can access the UI. Within this UI we have access to MLflow features.

In addition to the backend URI, we can also add an artifact root directory where we store the artifacts for runs, this is done by adding a --default-artifact-root paramater:

```bash

   mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

    ### MLflow Tracking Client API:

    In addition to the UI, an interface that is introduced in the course and used to automate processes is the Tracking API. Initialized through:
    ```python
    from mlflow.tracking import MlflowClient

    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
```

### Creating new Experiments:
We create an experiment in the top left corner of the UI. (In this instance nyc-taxi-experiment).

Using the Python API we use client.create_experiment("nyc-taxi-experiment").

### Tracking Single Experiment Runs with Mlflow in a Jupyter notebook or Python file:
In order to track experiment runs, we first initialize the mlflow experiment using the code:

``bash
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")
```

where we set the tracking URI and the current experiment name. In case the experiment does not exist, it will be automatically created.

We can then track a run, we'll use this simple code snippet as a starting point:

```bash
alpha = 0.01

lr = Lasso(alpha)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

mean_squared_error(y_val, y_pred, squared=False)
```

We initialize the run using

```bash
    with mlflow.start_run():
```
