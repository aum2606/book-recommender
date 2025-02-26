import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from content_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender
from hybrid_recommender import HybridRecommender
import os

# Initialize the recommendation system
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'data.csv')
data_processor = DataProcessor(data_path)
data = data_processor.load_data(sample_size=30000)

# Preprocess text features
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

def plot_top_books():
    """Plot top N books by average rating."""
    # Filter books with minimum number of ratings
    min_ratings = 50
    top_books = data[data['num_ratings'] >= min_ratings].copy()
    
    # Sort by rating and get top N
    top_books = top_books.nlargest(10, 'rating')[['title', 'rating', 'num_ratings']]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_books, y='title', x='rating', hue='title', legend=False)
    plt.title(f'Top 10 Books by Rating (min {min_ratings} ratings)')
    plt.xlabel('Rating')
    plt.ylabel('Book Title')
    plt.tight_layout()
    
    # Save plot to a temporary file
    plt.savefig('temp_plot.png')
    plt.close()
    return 'temp_plot.png'

def get_content_recommendations(book_title):
    """Get content-based recommendations for a book title."""
    try:
        book_id = data[data['title'].str.contains(book_title, case=False)].iloc[0]['book_id']
        recommendations = content_recommender.get_recommendations(book_id)
        return recommendations[['title', 'authors', 'similarity_score']].to_string()
    except (IndexError, KeyError):
        return "Book not found. Please try another title."

def get_collaborative_recommendations(user_id):
    """Get collaborative recommendations for a user ID."""
    try:
        user_id = int(user_id)
        if user_id < 0 or user_id >= 100:  # We created 100 dummy users
            return "Invalid user ID. Please enter a number between 0 and 99."
        recommendations = collaborative_recommender.get_recommendations(user_id)
        return recommendations[['title', 'authors', 'predicted_rating']].to_string()
    except ValueError:
        return "Invalid user ID. Please enter a number between 0 and 99."

def get_hybrid_recommendations(user_id, book_title):
    """Get hybrid recommendations based on both user ID and book title."""
    try:
        user_id = int(user_id)
        if user_id < 0 or user_id >= 100:
            return "Invalid user ID. Please enter a number between 0 and 99."
        
        book_id = data[data['title'].str.contains(book_title, case=False)].iloc[0]['book_id']
        recommendations = hybrid_recommender.get_recommendations(user_id, book_id)
        return recommendations[['title', 'authors', 'weighted_score']].to_string()
    except (IndexError, KeyError):
        return "Book not found. Please try another title."
    except ValueError:
        return "Invalid user ID. Please enter a number between 0 and 99."

# Create the Gradio interface
with gr.Blocks(title="Book Recommendation System") as demo:
    gr.Markdown("# Book Recommendation System")
    
    with gr.Tab("Top Books"):
        plot_button = gr.Button("Show Top 10 Books")
        plot_output = gr.Image(type="filepath")
        plot_button.click(plot_top_books, outputs=plot_output)
    
    with gr.Tab("Content-Based Recommendations"):
        gr.Markdown("Get recommendations based on book similarity")
        book_input = gr.Textbox(label="Enter a book title")
        content_output = gr.Textbox(label="Recommendations")
        gr.Button("Get Recommendations").click(
            get_content_recommendations, 
            inputs=book_input, 
            outputs=content_output
        )
    
    with gr.Tab("Collaborative Recommendations"):
        gr.Markdown("Get recommendations based on user ratings (Enter ID between 0-99)")
        user_input = gr.Textbox(label="Enter user ID")
        collab_output = gr.Textbox(label="Recommendations")
        gr.Button("Get Recommendations").click(
            get_collaborative_recommendations, 
            inputs=user_input, 
            outputs=collab_output
        )
    
    with gr.Tab("Hybrid Recommendations"):
        gr.Markdown("Get recommendations based on both user ratings and book similarity")
        hybrid_user_input = gr.Textbox(label="Enter user ID")
        hybrid_book_input = gr.Textbox(label="Enter a book title")
        hybrid_output = gr.Textbox(label="Recommendations")
        gr.Button("Get Recommendations").click(
            get_hybrid_recommendations, 
            inputs=[hybrid_user_input, hybrid_book_input], 
            outputs=hybrid_output
        )

if __name__ == "__main__":
    demo.launch()
