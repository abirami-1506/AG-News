# 📰 AG News Topic Classification: RNN vs Transformers

Comparative study of **five neural architectures** for four-class news topic classification on the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) benchmark: three recurrent models trained from scratch (**Simple RNN, LSTM, GRU**) and two pretrained Transformers fine-tuned for the task (**BERT-base, DistilBERT**).

> MSc Data Science · MAT339 – Transformer Models & Sequence Learning · SASTRA Deemed University

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-red) ![HuggingFace](https://img.shields.io/badge/🤗_Transformers-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Problem
Given a news title + description, classify it as **World, Sports, Business or Sci/Tech**.

## 🧪 Setup
- Full official data: 108,000 train / 12,000 validation (split from the official train set) / 7,600 test articles
- 5 epochs per model; checkpoint with the lowest validation loss restored before testing
- Metrics: accuracy, macro F1, per-class precision/recall/F1, confusion matrices, training and inference time, parameter count
- Single random seed, so small differences should be read cautiously

## 📊 Results (7,600-article test set)
| Model | Accuracy | Macro F1 |
|---|---|---|
| **BERT-base** | **0.9486** | **0.9486** |
| DistilBERT | 0.9464 | 0.9464 |
| LSTM | 0.9212 | 0.9209 |
| GRU | 0.9211 | 0.9208 |
| Simple RNN | 0.9121 | 0.9118 |

**Key findings**
- BERT is the most accurate; DistilBERT uses ~61.2% of BERT's parameters and about half its training/inference time for only ~0.21 pp lower accuracy.
- Transformers beat recurrent models by ~2.5 pp but need substantially more compute.
- *Business* and *Sci/Tech* are the most frequently confused classes across all models.

## 📁 Repository structure
```
├── notebooks/AG_News_RNN_vs_Transformers.ipynb   # full, runnable experiment
├── report/AG_News_Case_Study_Report.pdf          # complete written report
├── requirements.txt
└── README.md
```

## 🚀 How to run
**Kaggle (recommended):** upload the notebook, set *Accelerator → GPU* and *Internet → On*, then *Run All*.

**Locally:**
```bash
git clone https://github.com/<your-username>/ag-news-rnn-vs-transformers.git
cd ag-news-rnn-vs-transformers
pip install -r requirements.txt
jupyter notebook notebooks/AG_News_RNN_vs_Transformers.ipynb
```
A CUDA GPU is strongly recommended for BERT/DistilBERT. The dataset and pretrained weights download automatically from Hugging Face.

## 🛠 Tech stack
Python · PyTorch · Hugging Face Transformers & Datasets · scikit-learn · pandas · NumPy · Matplotlib

## ⚠️ Limitations
Single seed, fixed 5-epoch budget, limited hyperparameter tuning, and English-only news text.

## 👩‍💻 Author
**Abirami B** – M.Sc. Data Science, SASTRA Deemed University

## 📄 License
MIT
