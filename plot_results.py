import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def load_metrics(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metrics():
    files = [
        ("TF-IDF + LR", RESULTS_DIR / "tfidf_metrics.json"),
        ("Word2Vec Mean", RESULTS_DIR / "word2vec_metrics.json"),
        ("FastText Mean", RESULTS_DIR / "fasttext_metrics.json"),
    ]

    names, accs, f1s = [], [], []
    for name, path in files:
        data = load_metrics(path)
        names.append(name)
        accs.append(float(data["accuracy"]))
        f1s.append(float(data["macro_f1"]))

    x = range(len(names))
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(x, accs, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.xticks(list(x), names, rotation=20)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.bar(x, f1s, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.xticks(list(x), names, rotation=20)
    plt.title("Macro-F1 Comparison")
    plt.ylabel("Macro-F1")

    plt.tight_layout()
    out_path = RESULTS_DIR / "model_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved comparison plot to: {out_path}")


if __name__ == "__main__":
    plot_metrics()
