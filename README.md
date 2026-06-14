# AI-Based Skin Lesion Classification

## Overview

This project presents a hybrid deep learning and machine learning framework for multiclass skin lesion classification.

The system combines:

- MobileNetV2 feature extraction
- Clinical metadata
- XGBoost classification

to classify skin lesions into 11 disease classes.

## Dataset

MILK10k Dataset

## Technologies

- Python
- TensorFlow
- MobileNetV2
- XGBoost
- Streamlit

## Classes

AKIEC
BCC
BEN_OTH
BKL
DF
INF
MAL_OTH
MEL
NV
SCCKA
VASC

## Run

```bash
pip install -r requirements.txt
streamlit run app.py