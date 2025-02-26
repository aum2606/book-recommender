from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class ContentBasedRecommender:
    def __init__(self):
        """Initialize content-based recommender."""
        self.tfidf_matrix = None
        self.books_data = None

    def fit(self, tfidf_matrix, books_data):
        """
        Fit the recommender with TF-IDF matrix and book data.
        
        Args:
            tfidf_matrix: TF-IDF matrix of book features
            books_data: DataFrame containing book information
        """
        self.tfidf_matrix = tfidf_matrix
        self.books_data = books_data.copy()  # Make a copy to avoid warnings

    def get_recommendations(self, book_id, n_recommendations=5):
        """
        Get book recommendations based on content similarity.
        
        Args:
            book_id: ID of the book to base recommendations on
            n_recommendations: Number of recommendations to return
        
        Returns:
            DataFrame with recommended books
        """
        try:
            # Get the index of the book
            book_idx = self.books_data[self.books_data['book_id'] == book_id].index[0]
            
            # Calculate similarity scores
            sim_scores = cosine_similarity(
                self.tfidf_matrix[book_idx:book_idx+1], 
                self.tfidf_matrix
            ).flatten()
            
            # Get indices of top similar books
            similar_indices = np.argsort(sim_scores)[::-1][1:n_recommendations+1]
            
            # Create recommendations dataframe
            recommendations = []
            for idx in similar_indices:
                book_info = self.books_data.iloc[idx]
                recommendations.append({
                    'title': book_info['title'],
                    'authors': book_info['authors'],
                    'similarity_score': sim_scores[idx]
                })
            
            return pd.DataFrame(recommendations)
            
        except (IndexError, KeyError):
            return pd.DataFrame(columns=['title', 'authors', 'similarity_score'])

    def get_similar_books_by_features(self, features, n_recommendations=5):
        """
        Get recommendations based on text features.
        
        Args:
            features: Text features to base recommendations on
            n_recommendations: Number of recommendations to return
        """
        try:
            # Transform input features
            feature_matrix = self.tfidf.transform([features])
            
            # Calculate similarity
            sim_scores = cosine_similarity(feature_matrix, self.tfidf_matrix).flatten()
            
            # Get indices of top similar books
            similar_indices = np.argsort(sim_scores)[::-1][:n_recommendations]
            
            # Create recommendations dataframe
            recommendations = []
            for idx in similar_indices:
                book_info = self.books_data.iloc[idx]
                recommendations.append({
                    'title': book_info['title'],
                    'authors': book_info['authors'],
                    'similarity_score': sim_scores[idx]
                })
            
            return pd.DataFrame(recommendations)
            
        except Exception:
            return pd.DataFrame(columns=['title', 'authors', 'similarity_score'])
