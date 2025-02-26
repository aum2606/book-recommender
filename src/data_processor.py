import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class DataProcessor:
    def __init__(self, file_path):
        """Initialize DataProcessor with the path to the dataset."""
        self.file_path = file_path
        self.data = None
        self.train_data = None
        self.test_data = None
        self.tfidf_matrix = None
        self.tfidf = None

    def load_data(self, sample_size=None):
        """Load and optionally sample the dataset."""
        self.data = pd.read_csv(self.file_path)
        if sample_size:
            self.data = self.data[:sample_size]
            
        # Rename columns to match our code
        self.data = self.data.rename(columns={
            'bookId': 'book_id',
            'author': 'authors',
            'numRatings': 'num_ratings'
        })
        
        # Add a user_id column for collaborative filtering
        self.data['user_id'] = self.data.index % 100  # Create 100 dummy users
        
        return self.data

    def preprocess_text_features(self, text_columns):
        """Create TF-IDF features from text columns."""
        # Fill NaN values with empty string for each column
        text_data = self.data[text_columns].fillna('')
        
        # Combine all text columns
        combined_text = text_data.agg(' '.join, axis=1)
        
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(combined_text)
        return self.tfidf_matrix

    def split_data(self, test_size=0.2):
        """Split data into training and testing sets."""
        self.train_data, self.test_data = train_test_split(
            self.data, test_size=test_size, random_state=42
        )
        return self.train_data, self.test_data

    def get_user_item_ratings(self):
        """Get user-item ratings matrix."""
        return pd.pivot_table(
            self.data,
            values='rating',
            index='user_id',
            columns='book_id',
            fill_value=0
        )
