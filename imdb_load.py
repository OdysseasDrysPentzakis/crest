import os
import pandas as pd
from pathlib import Path
import wget
import tarfile

def download_and_extract_imdb():
    # Create directories
    base_dir = Path('data/imdb')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset if not exists
    tarfile_path = base_dir / 'aclImdb_v1.tar.gz'
    if not tarfile_path.exists():
        print("Downloading IMDB dataset...")
        wget.download(
            'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
            str(tarfile_path)
        )
    
    # Extract if not already extracted
    if not (base_dir / 'aclImdb').exists():
        print("\nExtracting dataset...")
        with tarfile.open(tarfile_path) as tar:
            tar.extractall(base_dir)

def read_imdb_split(split_dir):
    texts = []
    labels = []
    
    for label_dir in ['pos', 'neg']:
        label = 1 if label_dir == 'pos' else 0
        dir_path = os.path.join(split_dir, label_dir)
        
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    texts.append(text)
                    labels.append(label)
    
    return pd.DataFrame({'text': texts, 'label': labels})

def main():
    # Get the absolute path of the current script
    current_dir = Path.cwd()
    
    # Download and extract dataset
    download_and_extract_imdb()
    
    # Set up paths
    base_path = current_dir / 'data/imdb/aclImdb'
    train_path = base_path / 'train'
    test_path = base_path / 'test'
    output_path = base_path / 'processed'
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    print("Processing training data...")
    train_df = read_imdb_split(train_path)
    
    print("Processing test data...")
    test_df = read_imdb_split(test_path)
    
    print("Saving processed files...")
    train_df.to_parquet(output_path / 'train.parquet')
    test_df.to_parquet(output_path / 'test.parquet')
    
    print(f"Processed dataset saved to {output_path}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    main()
