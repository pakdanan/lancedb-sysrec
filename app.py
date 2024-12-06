import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import lancedb
import pyarrow as pa
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# 1. Load the MovieLens Dataset
movies = pd.read_csv("movies.csv")

tags = pd.read_csv('tags.csv')

# Aggregate tags for each movie
tags_grouped = tags.groupby('movieId')['tag'].apply(
    lambda x: ' '.join(x)).reset_index()

# Merge tags with movies
movies = movies.merge(tags_grouped, how='left',
                      left_on='movieId', right_on='movieId')
movies['tag'] = movies['tag'].fillna('')  # Fill NaN tags with empty strings

movies['genres'] = movies['genres'].str.replace(
    '|', ' ')  # Replace '|' with spaces

# Combine title, genres and tags into a single feature
movies['features'] = movies['title'] + ' ' + \
    movies['genres'] + ' ' + movies['tag']

# 2. Vectorize Movie Features Using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
movie_vectors = vectorizer.fit_transform(movies["features"]).toarray()

# Normalize vectors for better similarity calculation
movie_vectors = normalize(movie_vectors)

# 3. Store Movies and Vectors in LanceDB
# Connect to LanceDB
db = lancedb.connect("./lancedb_movies")

# Check if the table exists
if "movies" not in db.table_names():
    print("Creating the 'movies' table...")
    # Prepare data for LanceDB
    data = [
        {
            "movieId": row.movieId,
            "title": row.title,
            "vector": vector.tolist(),
        }
        for row, vector in zip(movies.itertuples(), movie_vectors)
    ]

    # Create LanceDB table with PyArrow Table
    arrow_table = pa.Table.from_pylist(data)
    table = db.create_table("movies", data=arrow_table)
else:
    print("The 'movies' table already exists.")
    table = db.open_table("movies")  # Open the existing table

# 4. Define API Endpoint to Get Recommendations


@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie_title = request.json.get('title')

    # Check if the movie exists
    selected_movie = movies[movies["title"] == movie_title]
    if selected_movie.empty:
        return jsonify({"error": "Movie not found"}), 404

    # Get the vector for the selected movie
    selected_row = selected_movie.iloc[0]
    movie_vector = movie_vectors[selected_row.name]

    # Query LanceDB for similar movies
    results = table.search(movie_vector.tolist()).limit(11).to_pandas()

    # Exclude the first result (index=0)
    filtered_results = results.iloc[1:]

    # Return the recommendations as a JSON response
    recommended_movies = filtered_results["title"].tolist()
    return jsonify({"recommendations": recommended_movies})


# 5. Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
