# Sentiment Analysis — NLP & Deep Learning

## Overview
3-class sentiment classification system (Positive, 
Negative, Neutral) using NLP, deep learning and 
transformer models on Financial News data.

## Tech Stack
- Python, PyTorch, RoBERTa, LoRA
- LSTM, GRU, NLTK, Word2Vec
- FastAPI, Gradio, MLflow

## Model Performance
| Model | F1 Score |
|---|---|
| BiLSTM | 0.81 |
| BiGRU | 0.84 |
| RoBERTa + LoRA | 0.93 ✅ Best |

## Key Highlights
- Trained on 5,307 financial news samples
- Fine-tuned RoBERTa with LoRA — best F1: 0.93
- Real-time prediction via FastAPI
- Gradio UI with single & batch prediction
- MLflow experiment tracking included
- 3 models compared and benchmarked

## Project Structure
├── app.py
├── gredio.py
├── datacleaning.ipynb
├── tf-idm lstm.ipynb
├── word2vec gru.ipynb
├── saved_models/
├── outputs/
├── mlruns/
└── requirements.txt
