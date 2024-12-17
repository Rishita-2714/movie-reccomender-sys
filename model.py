import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g., pd.read_csv)

# Load data from the 'Model' directory
movies = pd.read_csv('tmdb_5000_movies.xls')
credits = pd.read_csv('tmdb_5000_credits.xls')

# Merge the datasets on the 'title' column
movies = movies.merge(credits, on='title')

# Select necessary columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Handle missing data
movies.dropna(inplace=True)

import ast

# Function to extract 'name' values from JSON-like strings
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Apply the function to 'genres' and 'keywords' columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Extract top 3 cast members
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert3)

# Extract the director's name
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Remove spaces from the names
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# Split 'overview' into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine all relevant columns into a single 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new DataFrame with only 'movie_id', 'title', and 'tags'
new = movies[['movie_id', 'title', 'tags']]

# Use .loc to modify the 'tags' column to avoid SettingWithCopyWarning
new.loc[:, 'tags'] = new['tags'].apply(lambda x: " ".join(x))

# Vectorize the 'tags' column
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

# Recommendation function
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

# Example recommendation
recommend('Gandhi')

# Save the processed data and similarity matrix for later use
import pickle
pickle.dump(new, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
