# 20 Newsgroups Text Classifier

A text classification project using scikit-learn's 20 Newsgroups dataset and a TF-IDF + Logistic Regression pipeline.

## Dataset

Uses a 4-category subset of the 20 Newsgroups dataset:

- alt.atheism
- talk.religion.misc
- comp.graphics
- sci.space

The dataset is downloaded automatically via `fetch_20newsgroups`. 

## Project structure

- `notebooks/newsgroups_model.ipynb` – exploration and notebook walkthrough
- `src/train.py` – standalone training script
- `outputs/confusion_matrix.png` – confusion matrix
- `outputs/newsgroups_text_model.joblib` – saved text classification model

## Setup

```bash
python -m venv venv        # or: python3 -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Run

```bash
python src/train.py
```