# Late Payment Prediction
This project demonstrates a late payment prediction system featuring model training, feature engineering, and deployment using a message queue architecture with RabbitMQ. The system processes new loan applications, predicts the likelihood of late payment using a pre-trained ensemble model, and outputs the prediction.

Link to demo video: https://drive.google.com/file/d/1pvQGtygxg1GSRJuYjiIDi2evK861q30g/view?usp=sharing

## Table of Contents

* [Repository Structure](#repository-structure)
* [System Architecture](#system-architecture)
    * [Training Phase](#training-phase)
    * [Prediction Phase (Deployment Workflow)](#prediction-phase-deployment-workflow)
* [Running the System](#running-the-system)
    * [Prerequisites](#prerequisites)
    * [1. Setup Python Environment & Data](#1-setup-python-environment--data)
    * [2. Setup RabbitMQ](#2-setup-rabbitmq)
    * [3. Train Model](#3-train-model)
    * [4. Prepare Sample New Applications](#4-prepare-sample-new-applications)
    * [5. Start the Consumer (Prediction Service)](#5-start-the-consumer-prediction-service)
    * [6. Start the Producer (Send New Data)](#6-start-the-producer-send-new-data)
* [Key Scripts Overview](#key-scripts-overview)

---

## Repository Structure

```bash
loan_default_deployment/
├── data/                     # Input CSVs for training & new applications
│   ├── customer.csv
│   ├── loan.csv
│   ├── state_region.csv
│   ├── job_mapping.xlsx      
│   └── new_applications.csv  # Sample new data for the producer
├── saved_model/              # Stores trained model, preprocessor, and params
│   ├── model.joblib
│   ├── preprocessor.joblib
│   └── feature_engineering_params.json
├── notebook_and_ppt/
│   ├── [Feature Engineering] Job cluster.ipynb
│   ├── 20250530 Late Payment Prediction.ipynb
│   ├── Architecture.png
│   ├── demo.mp4
│   └── Project Presentation.pptx
├── train_model.py            # Script to train and save model/preprocessor
├── producer.py               # Script to send new data to RabbitMQ
├── consumer_predict.py       # Script to make predictions from RabbitMQ data
├── requirements.txt          # Python libraries needed
└── README.md                 # This file
```
---

## System Architecture

The system is divided into two main phases: an offline Training Phase and an online Prediction Phase.

### Training Phase

1.  **Data Loading & Merging**: Loads raw data from multiple CSV files (`customer.csv`, `loan.csv`, `state_region.csv`, `job_mapping.xlsx`).
2.  **Feature Engineering**: Performs extensive feature engineering, including creating new variables, cleaning existing ones, and mapping categorical values. Outlier capping parameters and other derived values (e.g., average interest rates per grade) are calculated and saved.
3.  **Data Splitting**: Data is split temporally for training to mimic a realistic scenario.
4.  **Preprocessing**: A `ColumnTransformer` is defined to handle numerical (imputation, scaling) and categorical (imputation, one-hot/ordinal encoding) features.
5.  **Model Training**: A `WeightedVotingClassifier` (ensemble of LightGBM, CatBoost, XGBoost) is trained on the preprocessed data.
6.  **Artifact Saving**: The trained preprocessor, model, and feature engineering parameters are saved to disk (`.joblib` and `.json` files) for use in the prediction phase.

### Prediction Phase (Deployment Workflow)

![Architecture](https://github.com/user-attachments/assets/95b58570-075c-4851-aedd-ec458b5176d4)

This phase uses a message-driven architecture with RabbitMQ for asynchronous processing of new loan applications.

1.  **New Application Ingest (Producer)**:
    * A `producer.py` script reads new loan applications from a source (e.g., `new_applications.csv`).
    * Each application's raw data is formatted into a JSON message.
    * The JSON message is sent to a specific queue (`loan_application_queue`) in RabbitMQ.

2.  **Message Queuing (RabbitMQ)**:
    * RabbitMQ acts as a message broker, holding the incoming loan applications in the `loan_application_queue`.
    * This decouples the application submission from the prediction service.

3.  **Prediction Service (Consumer)**:
    * A `consumer_predict.py` script continuously listens to the `loan_application_queue`.
    * When a message (JSON loan application) is received, the Consumer:
        1.  **Loads Artifacts**: Loads the pre-trained `model.joblib`, `preprocessor.joblib`, and `feature_engineering_params.json`.
        2.  **Feature Engineering**: Applies the *exact same* feature engineering steps to the new application data as were used during training, utilizing the loaded `feature_engineering_params.json` for consistency (e.g., for outlier capping values, mappings).
        3.  **Data Preprocessing**: Transforms the engineered features using the loaded `preprocessor.joblib`.
        4.  **Prediction**: Feeds the preprocessed data into the loaded `model.joblib` to predict the loan default status and probability.

4.  **Output**:
    * The prediction result ("Late Payment" or "On time", along with probability) is outputted.

---

## Running the System

Follow these steps to set up and run the loan default prediction system:

### Prerequisites

* Python (3.8+)
* Docker (for RabbitMQ, recommended) or a local RabbitMQ installation.

### 1. Setup Python Environment & Data

1.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare Data:**
    * Place your input CSV files (`customer.csv`, `loan.csv`, `state_region.csv`) and `job_mapping.xlsx` (or `job_mapping.csv`) into the `data/` directory.
    * Ensure the `saved_model/` directory exists (the training script will create it if not).

### 2. Setup RabbitMQ

The easiest way to run RabbitMQ is via Docker:
```bash
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```
### 3. Train Model

Run the training script to process data, train the model, and save the necessary artifacts:
```bash
python train_model.py
```
This will save `model.joblib`, `preprocessor.joblib`, and `feature_engineering_params.json` into the saved_model/ directory.
Monitor the console for any errors during training.

### 4. Prepare Sample New Applications

- Create a CSV file named new_applications.csv inside the data/ directory.
- The first row should be a header with the raw feature names expected by producer.py (these are columns from your original merged data before extensive feature engineering in train_model.py).
- Add a few rows of sample new loan application data.
 
### 5. Start the Consumer (Prediction Service)

Open a new terminal in your project directory
```bash
python consumer_predict.py
```
The consumer will connect to RabbitMQ and wait for messages in the loan_application_queue.

### 6. Start the Producer (Send New Data)

Open a third terminal in your project directory:
```bash
python producer.py
```
The producer will read data from data/new_applications.csv, send each application as a message to RabbitMQ.
You should see output in the consumer's terminal as it receives messages, processes them, and prints prediction results.

## Key Scripts Overview

`train_model.py`: Responsible for the entire model training pipeline: data loading, preprocessing, feature engineering, model training, and artifact serialization.

`producer.py`: Simulates the arrival of new loan applications, sending them as messages to a RabbitMQ queue.

`consumer_predict.py`: Listens to the RabbitMQ queue, processes incoming loan applications using the saved model artifacts, and outputs predictions. This script performs the critical steps of feature engineering and preprocessing on new data, mirroring the training phase.

`requirements.txt`: Lists all Python dependencies required for the project.
