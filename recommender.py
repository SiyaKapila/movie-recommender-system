import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
movies = pd.read_csv("data/movies.csv")

# Clean genres
movies["genres"] = movies["genres"].fillna("")

# Convert genres into vectors
vectorizer = CountVectorizer(tokenizer=lambda x: x.split("|"))
genre_matrix = vectorizer.fit_transform(movies["genres"])

# Compute similarity
similarity = cosine_similarity(genre_matrix)

def recommend(movie_title):

    idx = movies[movies["title"] == movie_title].index

    if len(idx) == 0:
        print("Movie not found")
        return

    idx = idx[0]

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return movies["title"].iloc[movie_indices]


if __name__ == "__main__":

    movie = "Toy Story (1995)"

    recs = recommend(movie)

    print("Recommendations for:", movie)
    print(recs)
