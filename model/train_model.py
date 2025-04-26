import os
import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from .genre_model import GenreClassifier
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, MLB_PATH
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path=None):
    """Load and preprocess data using train/validation split"""
    try:
        # Load training data
        train_df = pd.read_csv(
            train_path,
            sep='\t',
            header=None,
            names=['id', 'title', 'plot', 'genres'],
            dtype={'genres': str}
        )
        
        # Clean and process genres
        train_df['genres'] = train_df['genres'].str.strip().str.split('|')
        
        # Get all unique genres
        all_genres = set()
        all_genres.update(*train_df['genres'].dropna())
        
        # Initialize and fit MLB
        mlb = MultiLabelBinarizer(classes=sorted(all_genres))
        y = mlb.fit_transform(train_df['genres'])
        
        # Split into train and validation sets (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            train_df[['plot']], 
            y,
            test_size=0.2,
            random_state=42
        )
        
        print("\nData Verification:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print("\nSample training row:")
        print(X_train.iloc[0])
        print("\nClasses recognized by MLB:")
        print(mlb.classes_)
        
        return X_train, y_train, X_val, y_val, mlb
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def train_and_save_model():
    try:
        # Load data
        X_train, y_train, X_test, y_test, mlb = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
        
        # Initialize and train model
        model = GenreClassifier()
        model.train(X_train, y_train)
        
        # Evaluate - now returns a single float score
        score = model.evaluate(X_test, y_test)
        print(f"\nValidation F1 Score: {score:.4f}")  # Now works with float
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        with open(MLB_PATH, 'wb') as f:
            pickle.dump(mlb, f)
            
        print(f"\nModel saved to {MODEL_PATH}")
        return model, mlb
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise