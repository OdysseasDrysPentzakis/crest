import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split

nltk.download('punkt')

def preprocess_data(csv_file):
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Initial samples: {len(df)}")
    
    # Clean text
    df['text'] = df['text'].apply(lambda x: re.sub('<[^<]+?>', '', str(x)))  # Handle non-string values
    df['text'] = df['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))
    df['text'] = df['text'].str.lower().str.strip()
    
    # Clean labels
    df['label'] = df['label'].str.strip().str.lower()
    valid_labels = ['positive', 'negative']
    
    # Remove invalid labels
    df = df[df['label'].isin(valid_labels)]
    print(f"Samples after label cleaning: {len(df)}")
    
    # Convert labels to numeric
    df['label'] = df['label'].map({'positive': 1, 'negative': 0})
    
    # Check for class balance
    class_counts = df['label'].value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    # Final validation
    if df['label'].isna().any():
        print("NaN values in labels after conversion:")
        print(df[df['label'].isna()])
        raise ValueError("Invalid labels detected after conversion")

    # Stratified split only if we have both classes
    if len(class_counts) == 2:
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['label']
        )
    else:
        print("Warning: Using non-stratified split due to missing classes")
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42
        )

    return train_df, test_df

if __name__ == "__main__":
    train_data, test_data = preprocess_data('imdb_reviews.csv')
    
    train_data.to_csv('imdb_train_preprocessed.csv', index=False)
    test_data.to_csv('imdb_test_preprocessed.csv', index=False)
    
    print("\nFinal dataset sizes:")
    print(f"Train: {len(train_data)} samples")
    print(f"Test: {len(test_data)} samples")
    print("Preview of training data:")
    print(train_data.head())
