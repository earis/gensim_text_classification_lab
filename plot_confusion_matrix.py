import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def generate_confusion_matrix(model_name: str):
    pred_path = RESULTS_DIR / f"{model_name}_predictions.json"
    if not pred_path.exists():
        print(f"No prediction file found for {model_name}. Expected: {pred_path}")
        print("Please rerun the model pipeline to generate predictions.")
        return

    with open(pred_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not rows:
        print(f"Prediction file is empty: {pred_path}")
        return

    true_labels = [item["true_label"] for item in rows]
    pred_labels = [item["pred_label"] for item in rows]
    labels = sorted(set(true_labels) | set(pred_labels))

    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved confusion matrix to: {out_path}")


if __name__ == "__main__":
    for model in ["tfidf", "word2vec", "fasttext"]:
        generate_confusion_matrix(model)
