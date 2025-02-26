import pandas as pd
import numpy as np

class HybridRecommender:
    def __init__(self, content_recommender, collaborative_recommender, 
                 content_weight=0.5):
        """
        Initialize hybrid recommender system.
        
        Args:
            content_recommender: Instance of ContentBasedRecommender
            collaborative_recommender: Instance of CollaborativeRecommender
            content_weight: Weight for content-based recommendations (0-1)
        """
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.content_weight = content_weight
        self.collab_weight = 1 - content_weight

    def get_recommendations(self, user_id, book_id, n_recommendations=5):
        """
        Get hybrid recommendations combining both approaches.
        
        Args:
            user_id: ID of the user to recommend for
            book_id: ID of the book to base content recommendations on
            n_recommendations: Number of recommendations to return
        
        Returns:
            DataFrame with recommended books and combined scores
        """
        # Get recommendations from both systems
        content_recs = self.content_recommender.get_recommendations(
            book_id, 
            n_recommendations=n_recommendations
        )
        
        collab_recs = self.collaborative_recommender.get_recommendations(
            user_id, 
            n_recommendations=n_recommendations
        )

        # Normalize scores
        content_recs['norm_score'] = (content_recs['similarity_score'] - 
                                    content_recs['similarity_score'].min()) / \
                                   (content_recs['similarity_score'].max() - 
                                    content_recs['similarity_score'].min())
        
        collab_recs['norm_score'] = (collab_recs['predicted_rating'] - 
                                    collab_recs['predicted_rating'].min()) / \
                                   (collab_recs['predicted_rating'].max() - 
                                    collab_recs['predicted_rating'].min())

        # Combine recommendations
        all_recs = pd.concat([
            content_recs.assign(source='content'),
            collab_recs.assign(source='collaborative')
        ])

        # Calculate weighted scores
        all_recs['weighted_score'] = np.where(
            all_recs['source'] == 'content',
            all_recs['norm_score'] * self.content_weight,
            all_recs['norm_score'] * self.collab_weight
        )

        # Get top recommendations
        final_recs = all_recs.sort_values('weighted_score', 
                                        ascending=False).drop_duplicates('title')
        return final_recs.head(n_recommendations)
