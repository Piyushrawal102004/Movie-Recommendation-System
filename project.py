import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
movies_data = pd.read_csv('movies.csv')

# Fill NaN values in selected features with empty strings
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director', 'title']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine the features into a single string
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                    movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + \
                    movies_data['director']

# Vectorize the combined features using TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate the cosine similarity
similarity = cosine_similarity(feature_vectors)

# Get the list of all movie titles (removing any potential NaNs)
list_of_all_titles = movies_data['title'].dropna().tolist()

# Input the movie name
movie_name = input('Enter your favourite movie name: ')

# Find close matches
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

if find_close_match:
    close_match = find_close_match[0]
    # Get the index of the movie
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]

    # Get similarity scores and sort them
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Print suggested movies
    print('Movies suggested for you: \n')
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data.loc[index, 'title']
        if i < 30:
            print(f"{i}. {title_from_index}")
            i += 1
else:
    print("No close match found for the movie name.")
