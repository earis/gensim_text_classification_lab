# Gensim Text Classification Lab

This project is a lightweight implementation plan and working skeleton for a Chinese news text classification experiment based on the People's Daily 1998 corpus.

## Goals

- parse the raw PeopleDaily texts
- build weak labels from the article title
- compare multiple text representations
- train TF-IDF + Logistic Regression baseline
- train Gensim Word2Vec and FastText representations
- report accuracy and Macro-F1 metrics

## Project structure

- `main.py` : entry point for the pipeline
- `src/common.py` : project paths and utilities
- `src/preprocess.py` : raw corpus parsing and text cleaning
- `src/train_models.py` : TF-IDF, Word2Vec and FastText experiments
- `data/` : generated processed data and evaluation results

## Quick start

```bash
cd gensim_text_classification_lab
python main.py
```

## Notes

- This is a simplified research skeleton for coursework use.
- The label assignment is based on keyword matching in the title, so it is a weak-label setup.
- The code intentionally keeps the pipeline easy to understand and extend.
