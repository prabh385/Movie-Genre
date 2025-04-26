import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class GenreClassifier:
    def __init__(self):
        """Initialize the genre classifier"""
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced'
            ),
            n_jobs=-1
        )
    
    def _simple_preprocess(self, text):
        """Simplified text preprocessing"""
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        return text
    
    def train(self, X_train, y_train):
        """
        Train the model
        Args:
            X_train: DataFrame containing 'plot' column
            y_train: Multi-label targets
        """
        if 'plot' not in X_train.columns:
            raise ValueError(f"'plot' column not found. Available columns: {X_train.columns.tolist()}")
        
        print("Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train['plot'].apply(self._simple_preprocess))
        
        print("Training classifier...")
        self.classifier.fit(X_train_vec, y_train)
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        Args:
            X_test: DataFrame containing 'plot' column
            y_test: True labels
        Returns:
            float: F1 micro score
        """
        if 'plot' not in X_test.columns:
            raise ValueError(f"'plot' column not found. Available columns: {X_test.columns.tolist()}")
            
        X_test_vec = self.vectorizer.transform(X_test['plot'].apply(self._simple_preprocess))
        y_pred = self.classifier.predict(X_test_vec)
        return f1_score(y_test, y_pred, average='micro')
    
    def predict(self, text):
        """Predict genres for new text"""
        if isinstance(text, str):
            text = [text]
        text_vec = self.vectorizer.transform([self._simple_preprocess(t) for t in text])
        return self.classifier.predict(text_vec)
    
    def save(self, path):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)