import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.common import PROCESSED_PATH
from src.label_rules import assign_label


def clean_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""

    token = token.replace("[", "").replace("]", "")
    if "/" in token:
        token = token.rsplit("/", 1)[0]
    token = token.strip()
    token = re.sub(r"^[^\u4e00-\u9fffA-Za-z0-9]+|[^\u4e00-\u9fffA-Za-z0-9]+$", "", token)
    return token


def parse_article_text(raw_line: str) -> Tuple[str, str]:
    line = raw_line.strip()
    if not line:
        return "", ""

    parts = line.split()
    if len(parts) < 2:
        return "", ""

    article_id = parts[0].split("/")[0]
    title_tokens = []
    body_tokens = []
    seen_title = False

    for token in parts[1:]:
        clean = clean_token(token)
        if not clean:
            continue
        if not seen_title:
            title_tokens.append(clean)
            seen_title = True
        else:
            body_tokens.append(clean)

    title = " ".join(title_tokens)
    body = " ".join(body_tokens)
    return article_id, title, body


def load_raw_articles(raw_path: str | Path) -> List[Dict[str, str]]:
    articles: List[Dict[str, str]] = []
    raw_path = Path(raw_path)

    with raw_path.open("r", encoding="utf-8") as f:
        current_id = None
        title_parts = []
        body_parts = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            first_token = line.split()[0] if line.split() else ""
            if first_token and first_token.endswith("/m"):
                if current_id is not None:
                    title = " ".join(title_parts).strip()
                    body = " ".join(body_parts).strip()
                    if title or body:
                        articles.append({"id": current_id, "title": title, "body": body})

                current_id = first_token.split("/")[0]
                title_parts = []
                body_parts = []

            tokens = line.split()[1:] if first_token and first_token.endswith("/m") else line.split()
            cleaned_tokens = [clean_token(t) for t in tokens]
            cleaned_tokens = [t for t in cleaned_tokens if t]

            if current_id is None:
                continue

            if not title_parts and cleaned_tokens:
                title_parts.extend(cleaned_tokens[:20])
                body_parts.extend(cleaned_tokens[20:])
            else:
                body_parts.extend(cleaned_tokens)

        if current_id is not None:
            title = " ".join(title_parts).strip()
            body = " ".join(body_parts).strip()
            if title or body:
                articles.append({"id": current_id, "title": title, "body": body})

    return articles


def build_processed_dataset(raw_path: str | Path, output_dir: str | Path) -> List[Dict[str, str]]:
    articles = load_raw_articles(raw_path)
    processed: List[Dict[str, str]] = []

    for art in articles:
        title = art.get("title", "")
        body = art.get("body", "")
        label = assign_label(title)
        if not label:
            continue
        if len((body or "").split()) < 10:
            continue

        processed.append({
            "id": art["id"],
            "title": title,
            "body": body,
            "label": label,
            "text": body,
        })

    output_path = Path(output_dir) / "processed_articles.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return processed


def split_dataset(articles, output_dir: str | Path):
    from sklearn.model_selection import train_test_split

    labels = [article["label"] for article in articles]
    train, temp = train_test_split(
        articles,
        test_size=0.4,
        random_state=42,
        stratify=labels,
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42,
        stratify=[item["label"] for item in temp],
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, chunk in {"train.jsonl": train, "val.jsonl": val, "test.jsonl": test}.items():
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return train, val, test
