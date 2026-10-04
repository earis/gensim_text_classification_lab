from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT.parent / "PeopleDaily1998-master" / "PeopleDaily1998-master" / "199801" / "199801" / "199801.txt"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROJECT_ROOT / "data" / "results"

for path in [PROCESSED_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)
