from data_processor import DataProcessor
from content_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender
from hybrid_recommender import HybridRecommender
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_top_books(data, n=10):
    """Plot top N books by average rating."""
    # Filter books with minimum number of ratings
    min_ratings = 50
    top_books = data[data['num_ratings'] >= min_ratings].copy()
    
    # Sort by rating and get top N
    top_books = top_books.nlargest(n, 'rating')[['title', 'rating', 'num_ratings']]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_books, y='title', x='rating', hue='title', legend=False)
    plt.title(f'Top {n} Books by Rating (min {min_ratings} ratings)')
    plt.xlabel('Rating')
    plt.ylabel('Book Title')
    plt.tight_layout()
    plt.show()

def main():
    # Initialize data processor with correct path
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'data.csv')
    data_processor = DataProcessor(data_path)
    
    # Load data (use first 30000 rows as in notebook)
    data = data_processor.load_data(sample_size=30000)
    
    # Add a user_id column for collaborative filtering (using index as dummy user_id)
    data['user_id'] = data.index % 100  # Create 100 dummy users
    
    # Plot top books
    plot_top_books(data)
    
    # Preprocess text features for content-based filtering
    text_features = ['title', 'authors', 'description']
    tfidf_matrix = data_processor.preprocess_text_features(text_features)
    
    # Initialize recommenders
    content_recommender = ContentBasedRecommender()
    collaborative_recommender = CollaborativeRecommender()
    hybrid_recommender = HybridRecommender(
        content_recommender,
        collaborative_recommender,
        content_weight=0.5
    )
    
    # Fit the recommenders
    content_recommender.fit(tfidf_matrix, data)
    collaborative_recommender.fit(
        data[['user_id', 'book_id', 'rating']], 
        data
    )
    
    # Example recommendations
    print("\nContent-based recommendations for first book:")
    print(content_recommender.get_recommendations(data.iloc[0]['book_id']))
    
    print("\nCollaborative recommendations for first user:")
    print(collaborative_recommender.get_recommendations(data.iloc[0]['user_id']))
    
    print("\nHybrid recommendations:")
    print(hybrid_recommender.get_recommendations(
        data.iloc[0]['user_id'],
        data.iloc[0]['book_id']
    ))

if __name__ == "__main__":
    main()
