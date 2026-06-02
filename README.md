# 🧠 Sentiment Analysis — NLP & Deep Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Overview
A 3-class sentiment classification system (**Positive, Negative, Neutral**)
built on Financial News data using NLP, deep learning, and transformer models.
Benchmarks BiLSTM, BiGRU, and RoBERTa with LoRA fine-tuning.

---

## 🏆 Model Performance

| Model | F1 Score |
|---|---|
| BiLSTM | 0.81 |
| BiGRU | 0.84 |
| **RoBERTa + LoRA** | **0.93 ✅ Best** |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9 |
| Deep Learning | PyTorch, RoBERTa, LoRA, LSTM, GRU |
| NLP | NLTK, Word2Vec, TF-IDF |
| Serving | FastAPI, Gradio |
| Experiment Tracking | MLflow |

---

## ✨ Key Highlights
- 📊 Trained on **5,307 financial news samples**
- 🤗 Fine-tuned **RoBERTa with LoRA** — best F1: **0.93**
- ⚡ Real-time prediction via **FastAPI** inference API
- 🖥️ **Gradio UI** with single & batch prediction support
- 📈 **MLflow** experiment tracking and model versioning
- 🔬 3 models benchmarked and compared

---

## 📁 Project Structure
sentiment-analysis-nlp/
├── app.py                  # FastAPI inference API
├── gradio_app.py           # Gradio UI for predictions
├── datacleaning.ipynb      # Data preprocessing pipeline
├── tf-idm_lstm.ipynb       # BiLSTM training notebook
├── word2vec_gru.ipynb      # BiGRU training notebook
├── saved_models/           # Trained model checkpoints
├── outputs/                # Evaluation results & plots
├── data.csv                # Raw dataset
├── data_cleaned.csv        # Cleaned dataset
└── requirements.txt        # Dependencies
---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/bhanuprakash272400/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI server
```bash
uvicorn app:app --reload
```
API live at: http://localhost:8000
Swagger docs: http://localhost:8000/docs

### 4. Run Gradio UI
```bash
python gradio_app.py
```
UI live at: http://localhost:7860

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction (max 50) |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The company reported record profits this quarter."}'
```

### Example Response
```json
{
  "text": "The company reported record profits this quarter.",
  "sentiment": "Positive",
  "confidence": 0.9341,
  "probabilities": {
    "Negative": 0.0312,
    "Neutral": 0.0347,
    "Positive": 0.9341
  }
}
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Financial News Sentiment Dataset |
| **Total Samples** | 5,307 |
| **Negative** | 590 |
| **Neutral** | 2,874 |
| **Positive** | 1,843 |

---

## 📬 Contact

**Bhanu Prakash Theertham**
📧 bhanu.theertham@gmail.com
🔗 [github.com/bhanuprakash272400](https://github.com/bhanuprakash272400)
