from pathlib import Path

from src.common import DATASET_PATH, PROCESSED_PATH, RESULTS_PATH
from src.preprocess import build_processed_dataset, split_dataset
from src.train_models import run_tfidf_baseline, run_word2vec_experiment, run_fasttext_experiment


def main() -> None:
    project_root = Path(__file__).resolve().parent
    raw_path = DATASET_PATH
    output_dir = PROCESSED_PATH
    results_dir = RESULTS_PATH

    print(f"Reading dataset from: {raw_path}")
    articles = build_processed_dataset(raw_path, output_dir)
    print(f"Total articles parsed: {len(articles)}")

    train_df, val_df, test_df = split_dataset(articles, output_dir)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n[1/3] TF-IDF + Logistic Regression")
    run_tfidf_baseline(train_df, val_df, test_df, results_dir)

    print("\n[2/3] Word2Vec averaged vector baseline")
    run_word2vec_experiment(train_df, val_df, test_df, results_dir)

    print("\n[3/3] FastText experiment")
    run_fasttext_experiment(train_df, val_df, test_df, results_dir)

    print("\nDone. Results saved to:", results_dir)


if __name__ == "__main__":
    main()
