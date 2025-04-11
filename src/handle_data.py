import pandas as pd
from pathlib import Path

def load_reviews(data_dir, eval=False, load_all=False):
    """
    Load text reviews from subdirectories.
    
    Expects folders named like 'real_positive', 'fake_negative', etc.
    Adjust folder names as needed.
    
    Parameters:
    - data_dir: Directory containing the data folders.
    - eval: If True, loads the evaluation set. Defaults to False (loads the training set).
    - load_all: If True, loads both training and evaluation sets. Defaults to False.
    """
    reviews, labels = [], []
    data_dir = Path(data_dir)
    
    if load_all:
        # Load both training and evaluation sets
        data_dirs = [data_dir / "train", data_dir / "eval"]
    else:
        # Load only the selected dataset (train or eval)
        data_dirs = [data_dir / ("eval" if eval else "train")]

    for data_path in data_dirs:
        for label in ["deceptive", "truthful"]:
            text_files = list((data_path / label).rglob("*.txt"))
            for file in text_files:
                reviews.append(file.read_text(encoding="utf-8"))
                labels.append(label)
    
    return pd.DataFrame({"review": reviews, "label": labels})
