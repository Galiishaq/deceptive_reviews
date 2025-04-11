import random
import shutil
import zipfile
import tempfile
import requests
import argparse
from pathlib import Path


def download_zipfile(url, save_directory, filename="downloaded_file.zip"):
    """
    Downloads a ZIP file from a given URL and saves it to a specified directory.

    Parameters:
        url (str): The URL of the ZIP file to download.
        save_directory (str or Path): The directory where the file should be saved.
        filename (str): The name to save the file as (default: 'downloaded_file.zip').

    Returns:
        Path: The path to the saved file.
    """
    # Convert save_directory to a Path object
    save_directory = Path(save_directory)
    
    # Ensure the save directory exists
    save_directory.mkdir(parents=True, exist_ok=True)
    
    # Full path for the saved file
    file_path = save_directory / filename
    
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with file_path.open('wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded and saved to {file_path}")
        return file_path
    else:
        raise Exception(f"Failed to download file: {response.status_code} {response.reason}")
    

def split_and_organize_data(source_dir, output_dir, train_ratio=0.8):
    """
    Splits the data into train and eval sets and organizes them by label.
    
    Args:
        source_dir (str): Path to the source data directory.
        output_dir (str): Path to the output directory.
        train_ratio (float): Proportion of data for training (default: 0.8).
    """
    random.seed(42)
    output_dir = Path(output_dir)
    for subset in ["train", "eval"]:
        (output_dir / subset).mkdir(parents=True, exist_ok=True)

    for label in ["deceptive", "truthful"]:
        files = list(Path(source_dir).rglob(f"{label}*/*/*.txt"))
        random.shuffle(files)
        train_cutoff = int(len(files) * train_ratio)

        for subset, subset_files in zip(["train", "eval"], [files[:train_cutoff], files[train_cutoff:]]):
            subset_dir = output_dir / subset / label
            subset_dir.mkdir(parents=True, exist_ok=True)
            for file in subset_files:
                shutil.copy(file, subset_dir / file.name)

    print(f"Data split and organized in '{output_dir}'.")

def process_zip_to_dataset(zip_file, output_dir="."):
    """
    Extracts a zip file, splits data into train/eval sets, and organizes it.
    
    Args:
        zip_file (str): Path to the zip file containing data.
        output_dir (str): Path to the output directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        split_and_organize_data(temp_dir, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset by downloading, extracting, and splitting data.")
    parser.add_argument("--url", type=str, default="https://myleott.com/op_spam_v1.4.zip", help="URL of the zip file to download.")
    parser.add_argument("--save_dir", type=str, default="data", help="Directory to save the downloaded and processed data.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of data for training.")
    parser.add_argument("--zip_name", type=str, default="deceptive_reviews.zip", help="Name of the downloaded zip file.")

    args = parser.parse_args()

    # Download the zip file
    zip_file_path = download_zipfile(args.url, args.save_dir, filename=args.zip_name)
    
    # Process the zip file into a dataset
    process_zip_to_dataset(zip_file_path, args.save_dir)

if __name__ == "__main__":
    main()
