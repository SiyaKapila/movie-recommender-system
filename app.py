import streamlit as st
from recommender import recommend
import pandas as pd

movies = pd.read_csv("data/movies.csv")

st.title("Movie Recommendation System")

movie_list = movies["title"].values

selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.write("Recommended Movies:")

    for movie in recommendations:
        st.write(movie)

