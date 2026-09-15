# TruthLens AI

### AI-Powered Fake News Detection

TruthLens AI is an end-to-end machine learning application that classifies news articles as **Fake** or **Real** using a fine-tuned **DistilBERT** transformer model.

The project follows a complete machine learning workflow — from dataset analysis and preprocessing to model training, evaluation, optimized inference, API development, containerization, and cloud deployment.

A traditional **TF-IDF + Logistic Regression** model was developed as a baseline and compared with the fine-tuned DistilBERT model. For production inference, the transformer model was converted to **ONNX** and dynamically quantized to **INT8**, significantly reducing its size and making CPU-based deployment practical.

The application provides an interactive **Streamlit frontend** connected to a **FastAPI REST API**, with the optimized model hosted on **Hugging Face** and the application services deployed on **Render**.

---

## Live Demo

Try the deployed application:

**[TruthLens AI — Live Demo](https://truthlens-ai-1z0g.onrender.com)**

The application provides an interactive interface where users can enter news text and receive:

- Fake or Real prediction
- Prediction confidence
- Fake probability
- Real probability

### Key Features

- **Machine Learning Classification** — Detects whether news content is likely Fake or Real.
- **DistilBERT Transformer** — Uses a fine-tuned DistilBERT model for text classification.
- **Baseline Comparison** — Compares transformer performance against a TF-IDF + Logistic Regression baseline.
- **Optimized Inference** — Uses ONNX Runtime with dynamic INT8 quantization for lightweight CPU inference.
- **REST API** — Provides predictions through a FastAPI backend.
- **Interactive Web Interface** — Built with Streamlit.
- **Containerized Deployment** — Backend is packaged using Docker.
- **Cloud Deployment** — Frontend and backend are deployed on Render, with the optimized model hosted on Hugging Face.
- **Automated Testing** — Includes API and prediction tests using pytest.

---

## Project Overview

TruthLens AI was developed as an end-to-end machine learning project covering the complete workflow from raw data to a deployed prediction service.

The project follows these stages:

1. **Dataset Analysis** — Compared the ISOT and WELFake datasets before selecting the dataset used for final training.
2. **Data Preprocessing** — Cleaned the selected dataset and prepared the text and labels for model training.
3. **Exploratory Data Analysis** — Examined label distribution, missing values, and article-length characteristics.
4. **Baseline Modeling** — Built a TF-IDF + Logistic Regression classifier as a traditional machine learning benchmark.
5. **Transformer Fine-Tuning** — Fine-tuned DistilBERT for binary fake-news classification.
6. **Model Evaluation** — Evaluated the models using accuracy, precision, recall, and F1 score.
7. **Inference Optimization** — Converted the trained transformer to ONNX and applied dynamic INT8 quantization.
8. **API Development** — Exposed the prediction pipeline through a FastAPI REST API.
9. **Containerization** — Packaged the backend using Docker.
10. **Cloud Deployment** — Deployed the frontend and backend on Render and hosted the optimized model on Hugging Face.

---

## System Architecture

The production system is organized into separate frontend, backend, and model-inference components:

```text
                         User
                           |
                           v
                +---------------------+
                |    Streamlit UI     |
                |      (Render)       |
                +----------+----------+
                           |
                           | HTTP POST
                           v
                +---------------------+
                |     FastAPI API     |
                |      (Render)       |
                +----------+----------+
                           |
                           v
                +---------------------+
                |     Prediction      |
                |       Pipeline      |
                +----------+----------+
                           |
                           v
                +---------------------+
                |    ONNX Runtime     |
                |  INT8 Quantized     |
                |     DistilBERT      |
                +----------+----------+
                           |
                           v
                +---------------------+
                |  Hugging Face Model |
                |      Repository     |
                +---------------------+
```

The frontend communicates with the backend through HTTP, while the backend handles tokenization, model inference, probability calculation, and prediction formatting.

The production model uses **ONNX Runtime with CPU execution**, allowing the application to run without requiring a GPU.

---

## Dataset

Two publicly available fake-news datasets were considered during the project:

1. **ISOT Fake News Dataset**
2. **WELFake Dataset**

The datasets were compared based on factors such as dataset size, diversity, and suitability for the classification task. The **WELFake Dataset** was selected for the final training pipeline.

The dataset contains the following relevant fields:

- `title`
- `text`
- `label`

After preprocessing, the final dataset used for model development contains:

**62,592 articles**

The dataset-selection analysis is documented in:

`docs/dataset_decision.md`

### Label Encoding

The WELFake labels used by the training pipeline are:

0 → Real
1 → Fake

---

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the structure and characteristics of the dataset before model training.

The analysis focused on:

- Distribution of Fake and Real news articles
- Missing values across the dataset
- Duplicate records
- Distribution of article and title lengths
- Basic text characteristics

These checks helped identify data-quality issues and provided a better understanding of the classification problem.

The EDA notebooks and generated analysis outputs are available in:

`notebooks/`

---

## Baseline Model

A traditional machine learning approach was implemented as a baseline for comparison with the transformer-based model.

The baseline pipeline uses:

**TF-IDF (Term Frequency–Inverse Document Frequency) + Logistic Regression**

TF-IDF converts the news text into numerical feature vectors based on the importance of words within the dataset. Logistic Regression then uses these features to classify each article as Fake or Real.

### Baseline Performance

The baseline model achieved approximately:

| Metric | Score |
|---|---:|
| Accuracy | 93.67% |

This baseline provides a useful benchmark for evaluating the improvement obtained from the fine-tuned DistilBERT model.

The baseline implementation is located in:

`src/models/`

---

## Transformer Model

For the final classification model, the project uses **DistilBERT**, a lightweight transformer architecture based on BERT.

The model was fine-tuned using the `distilbert-base-uncased` pretrained checkpoint for binary classification of news articles.

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | `distilbert-base-uncased` |
| Maximum Sequence Length | 256 tokens |
| Batch Size | 16 |
| Learning Rate | `2e-5` |
| Epochs | 2 |
| Train / Validation Split | 80 / 20 |
| Random State | 42 |
| Training Environment | Google Colab T4 GPU |

The model was trained on the preprocessed WELFake dataset using the Hugging Face Transformers training framework.

### Model Performance

The fine-tuned DistilBERT model achieved the following validation performance:

| Metric | Score |
|---|---:|
| Accuracy | 99.12% |
| Precision | 99.31% |
| Recall | 98.71% |
| F1 Score | 99.01% |

The results show a substantial improvement over the TF-IDF + Logistic Regression baseline.

The training and evaluation workflow is documented in:

`notebooks/`

---

## Inference Optimization

The original fine-tuned DistilBERT model was optimized for production deployment to reduce memory usage and make CPU-based inference practical.

The optimization pipeline consists of two stages:

1. **ONNX Conversion** — The trained transformer model was converted from the PyTorch-based format to ONNX.
2. **INT8 Quantization** — Dynamic INT8 quantization was applied to the ONNX model using ONNX Runtime.

### Model Size Reduction

| Model | Approximate Size |
|---|---:|
| Original ONNX Model | 267.9 MB |
| INT8 Quantized Model | 67.3 MB |

This reduced the model size by approximately **75%**.

The optimized model is hosted on the Hugging Face Hub and is downloaded by the backend when required.

### Production Inference

The production prediction pipeline uses:

```text
Input Text
    ↓
DistilBERT Tokenizer
    ↓
ONNX Runtime
    ↓
INT8 Quantized DistilBERT
    ↓
Logits
    ↓
Softmax Probabilities
    ↓
Fake / Real Prediction
```
---

## Prediction Pipeline

The prediction pipeline is responsible for loading the optimized model, processing input text, running inference, and returning the final classification result.

The pipeline performs the following steps:

1. **Load the tokenizer** associated with the fine-tuned DistilBERT model.
2. **Tokenize the input text** with truncation and a maximum sequence length of 256 tokens.
3. **Run ONNX inference** using ONNX Runtime.
4. **Convert model logits into probabilities** using the softmax function.
5. **Determine the predicted class** from the highest probability.
6. **Map the model class to the final label**:
   - `0 → Real`
   - `1 → Fake`
7. **Return the prediction, confidence, and class probabilities.**

### Prediction Response

A prediction returned by the backend follows this structure:

```json
{
  "prediction": "Fake",
  "confidence": 0.9995,
  "probabilities": {
    "fake": 0.9995,
    "real": 0.0005
  }
}
```

The inference implementation is located in:

`src/models/predict.py`

The prediction pipeline is shared by the API layer, keeping model inference separate from the web interface and API logic.

---

## REST API

TruthLens AI exposes the prediction functionality through a **FastAPI REST API**.

### API Endpoints

| Method | Endpoint   |         Description                  |
|--------|----------- |------------------------------------- |
| `GET`  | `/health`  | Checks whether the API is running    |
| `POST` | `/predict` | Classifies news text as Fake or Real |

### Run the API Locally

Start the FastAPI server with:

```bash
python -m uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Swagger API Documentation

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### Prediction Request

Send a `POST` request to `/predict` with the news text:

```json
{
  "text": "Scientists announced a new discovery after years of research."
}
```

### Example Response

```json
{
  "prediction": "Fake",
  "confidence": 0.98,
  "probabilities": {
    "fake": 0.98,
    "real": 0.02
  }
}
```

The API layer is implemented in:

`api/`

---

## Docker

The FastAPI backend is containerized using **Docker** for reproducible deployment.

The Docker image uses a lightweight Python 3.11 base image and installs only the dependencies required for production inference and API serving.

### Build the Docker Image

```bash
docker build -t truthlens-ai .
```

### Run the Container

```bash
docker run --name truthlens-api -p 8000:8000 truthlens-ai
```

The API will then be available at:

```text
http://localhost:8000
```

Swagger documentation is available at:

```text
http://localhost:8000/docs
```

The Docker configuration is defined in:

`Dockerfile`

---

## Testing

Automated tests are implemented using **pytest** to verify the API and prediction pipeline.

The test suite covers:

- Health endpoint
- Prediction endpoint
- Empty input validation
- Missing input validation
- Prediction label validation
- Prediction confidence validation
- Class probability validation
- Probability consistency

### Run the Tests

From the project root, run:

```bash
pytest -v
```

### Current Test Result

```text
8 passed
```

The tests are located in:

`tests/`

---

## Project Structure

```text
fake_news_detection/
|
+-- api/
|   +-- main.py
|   +-- routes.py
|   +-- schemas.py
|
+-- data/
|   +-- raw/
|   +-- processed/
|   +-- reports/
|
+-- docs/
|   +-- dataset_decision.md
|
+-- models/
|
+-- src/
|   +-- config/
|   |   +-- paths.py
|   |
|   +-- data/
|   |   +-- audit.py
|   |   +-- compare_datasets.py
|   |   +-- loader.py
|   |   +-- preprocess.py
|   |
|   +-- models/
|   |   +-- evaluate.py
|   |   +-- predict.py
|   |   +-- train_baseline.py
|   |   +-- train_transformer.py
|   |
|   +-- utils/
|   |   +-- helpers.py
|   |
|   +-- visualization/
|       +-- eda.py
|
+-- tests/
|   +-- test_api.py
|   +-- test_predict.py
|
+-- .dockerignore
+-- .gitignore
+-- Dockerfile
+-- LICENSE
+-- README.md
+-- requirements.txt
+-- requirements-frontend.txt
+-- app.py
```

The project is organized to keep data processing, model development, API functionality, testing, and frontend components separated.

Large datasets, generated artifacts, and trained model files are excluded from Git and are not stored directly in the repository.

---

## Technologies Used

### Programming & Data Processing

- Python
- Pandas
- NumPy
- Scikit-learn

### Machine Learning & NLP

- Hugging Face Transformers
- DistilBERT
- PyTorch
- ONNX
- ONNX Runtime

### Backend & API

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- Streamlit

### Deployment & Infrastructure

- Docker
- Render
- Hugging Face Hub

### Testing & Version Control

- Pytest
- Git
- GitHub

---

## Deployment Architecture

The application is deployed as separate frontend and backend services.

```text
                         User
                           |
                           v
              +-----------------------+
              |   Streamlit Frontend  |
              |        Render         |
              +-----------+-----------+
                          |
                          | HTTPS
                          v
              +-----------------------+
              |     FastAPI Backend   |
              |        Render         |
              +-----------+-----------+
                          |
                          v
              +-----------------------+
              |    ONNX Runtime       |
              |    INT8 DistilBERT    |
              |    CPU Inference      |
              +-----------+-----------+
                          |
                          v
              +-----------------------+
              |    Hugging Face Hub   |
              |  Optimized Model      |
              +-----------------------+
```

### Production Services

| Component |    Platform      |           Purpose              |
|-----------|----------------- |--------------------------------|
| Frontend  | Render           | Streamlit web application      |
| Backend   | Render           | FastAPI prediction API         |
| Model     | Hugging Face Hub | Hosts the optimized ONNX model |
| Inference | ONNX Runtime     | CPU-based model inference      |

The frontend communicates with the backend using HTTP requests. The backend loads the optimized model and performs inference using ONNX Runtime.

The production deployment does not require a GPU.

---

## Repository & Data Policy

Large datasets, generated artifacts, and trained model files are intentionally excluded from the Git repository.

The following directories are excluded through `.gitignore`:

```text
data/raw/
data/processed/
data/reports/
models/
venv/
```

---

## Future Improvements

Potential improvements to the current system include:

- Testing the model on additional external datasets to evaluate generalization.
- Improving robustness against unseen news sources and writing styles.
- Adding model explainability techniques to help users understand predictions.
- Calibrating prediction confidence for more reliable probability estimates.
- Monitoring model performance and prediction behavior after deployment.
- Adding authentication and rate limiting to the API.
- Exploring further model compression and optimization techniques for more resource-efficient deployment.

---

## Author

**Esther**

TruthLens AI - Fake News Detection Project