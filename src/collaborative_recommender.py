import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    def __init__(self):
        """Initialize collaborative filtering recommender."""
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.books_data = None

    def fit(self, ratings_data, books_data):
        """
        Fit the collaborative filtering model.
        
        Args:
            ratings_data: DataFrame with columns [user_id, book_id, rating]
            books_data: DataFrame with book information
        """
        self.books_data = books_data.copy()  # Make a copy to avoid warnings
        
        # Create user-item matrix
        self.user_item_matrix = pd.pivot_table(
            ratings_data,
            values='rating',
            index='user_id',
            columns='book_id',
            fill_value=0
        )
        
        # Calculate item-item similarity
        self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        # Create a mapping of matrix indices to book_ids
        self.book_indices = {book_id: idx for idx, book_id in 
                           enumerate(self.user_item_matrix.columns)}
        self.inverse_book_indices = {idx: book_id for book_id, idx in 
                                   self.book_indices.items()}

    def get_recommendations(self, user_id, n_recommendations=5):
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id: ID of the user to recommend for
            n_recommendations: Number of recommendations to return
        
        Returns:
            DataFrame with recommended books
        """
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame(columns=['title', 'authors', 'predicted_rating'])
            
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Calculate predicted ratings using weighted sum
        weighted_sums = np.dot(self.similarity_matrix, user_ratings)
        similarity_sums = np.sum(np.abs(self.similarity_matrix), axis=1)
        
        # Avoid division by zero
        similarity_sums[similarity_sums == 0] = 1e-10
        predicted_ratings = weighted_sums / similarity_sums
        
        # Get indices of top recommended books
        user_rated = user_ratings > 0
        unrated_indices = np.where(~user_rated)[0]
        
        if len(unrated_indices) == 0:
            return pd.DataFrame(columns=['title', 'authors', 'predicted_rating'])
            
        unrated_predictions = predicted_ratings[unrated_indices]
        recommended_indices = unrated_indices[np.argsort(unrated_predictions)[::-1][:n_recommendations]]
        
        # Get book IDs for recommended indices
        recommended_book_ids = [self.inverse_book_indices[idx] for idx in recommended_indices]
        
        # Get book details and predicted ratings
        recommendations = []
        for idx, book_id in zip(recommended_indices, recommended_book_ids):
            book_info = self.books_data[self.books_data['book_id'] == book_id].iloc[0]
            recommendations.append({
                'title': book_info['title'],
                'authors': book_info['authors'],
                'predicted_rating': predicted_ratings[idx]
            })
        
        return pd.DataFrame(recommendations)
