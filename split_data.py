import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = "spectrogramData"  # Path to the root directory containing unsplit data
SPLIT_DIR = "split_data"  # Directory to store split data
TEST_SPLIT = 0.2  # Fraction of data to use for testing

def prepare_split_data(data_dir, split_dir, test_split):
    """Splits the dataset into train and test folders."""
    # Create directories for split data
    split_dir = Path(split_dir)
    train_dir = split_dir / "train"
    test_dir = split_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for category in os.listdir(data_dir):
        category_path = Path(data_dir) / category
        if not category_path.is_dir():
            continue

        # Get all files in the category folder
        files = list(category_path.glob("*"))
        train_files, test_files = train_test_split(files, test_size=test_split, random_state=42)

        # Move files into respective train and test folders
        for train_file in train_files:
            (train_dir / category).mkdir(parents=True, exist_ok=True)
            shutil.copy(train_file, train_dir / category / train_file.name)

        for test_file in test_files:
            (test_dir / category).mkdir(parents=True, exist_ok=True)
            shutil.copy(test_file, test_dir / category / test_file.name)

# Prepare the split dataset
prepare_split_data(DATA_DIR, SPLIT_DIR, TEST_SPLIT)