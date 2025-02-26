# Book Recommendation System

A sophisticated book recommendation system that uses multiple recommendation methods to suggest books to users. The system provides personalized book recommendations using various techniques including content-based filtering, collaborative filtering, and a hybrid approach.

## Features

- **Content-Based Filtering**: 
  - Recommends books based on similarity in content (title, author, description)
  - Uses TF-IDF vectorization for text feature extraction
  - Considers book attributes like genre, authors, and description

- **Collaborative Filtering**: 
  - Recommends books based on user ratings and behavior
  - Uses cosine similarity to find similar user preferences
  - Handles new users and items effectively

- **Hybrid Recommendations**: 
  - Combines both content-based and collaborative filtering
  - Weighted approach to balance both methods
  - Better accuracy through combined predictions

- **Interactive UI**: 
  - Built with Gradio for easy interaction
  - Visual display of top rated books
  - Simple interface for getting recommendations
  - Real-time recommendation generation

## Dataset

The system uses a comprehensive book dataset with the following information:
- Book metadata (ID, title, author, description)
- User ratings and interactions
- Additional book attributes (genres, number of ratings)

Required columns in data.csv:
```
bookId      : Unique identifier for each book
title       : Book title
author      : Book author(s)
rating      : Average rating (0-5)
description : Book description
isbn        : ISBN number
genres      : Book genres
numRatings  : Number of ratings received
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/book-recommendation-system.git
cd book-recommendation-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your dataset:
- Place your book dataset (data.csv) in the `data` directory
- Ensure the dataset follows the required column structure

## Usage

1. Start the web interface:
```bash
python src/app.py
```

2. Access the interface at `http://localhost:7861`

3. Use the different tabs for recommendations:
- **Top Books**: View the highest rated books
- **Content-Based**: Enter a book title to find similar books
- **Collaborative**: Enter a user ID (0-99) to get personalized recommendations
- **Hybrid**: Combine both approaches by providing both user ID and book title

## Project Structure

```
book-recommendation-system/
├── data/               # Dataset directory
│   └── data.csv       # Book dataset (not included in repo)
├── src/               # Source code
│   ├── app.py         # Gradio web interface
│   ├── data_processor.py       # Data loading and preprocessing
│   ├── content_recommender.py  # Content-based filtering
│   ├── collaborative_recommender.py  # Collaborative filtering
│   └── hybrid_recommender.py   # Hybrid approach
├── requirements.txt    # Python dependencies
├── LICENSE            # MIT license
└── README.md          # Project documentation
```

## Implementation Details

### Data Preprocessing
- Text cleaning and normalization
- TF-IDF vectorization for text features
- Handling missing values
- Creating user-item interaction matrix

### Recommendation Algorithms
1. **Content-Based**:
   - TF-IDF for text feature extraction
   - Cosine similarity for book matching
   - Title and description analysis

2. **Collaborative**:
   - User-item matrix creation
   - Similarity computation
   - Rating prediction

3. **Hybrid**:
   - Weighted combination of both methods
   - Configurable weighting system
   - Score normalization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
