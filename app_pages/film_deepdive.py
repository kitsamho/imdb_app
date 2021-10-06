import pandas as pd
import os
import json
import streamlit as st
import plotly.express as px
from app_dependencies.semantics import get_embeddings, getAllNounChunks, visualise_nounchunks
from app_dependencies.visuals import plotly_streamlit_layout, plotly_streamlit_texts
from app_dependencies.actor_deepdive import get_film_image


def write(df, df_actor_all):
    st.title('Film Deepdive')

    st.write(df_actor_all.head())

    df = df[['tmdb_id', 'movie', 'overview', 'genres', 'genres_new', 'poster_path',
                       'popularity', 'release_year', 'vote_average', 'budget', 'revenue']]


    df_film = df.explode('genres')
    df_film = df_film.dropna(subset=['genres'])
    st.text("")
    st.text("")
    st.write('Select genres to include:')
    c1,c2, c3 = st.columns((1,2,2))

    x = {genre:c1.checkbox(genre) for genre in df_film.genres.unique()}

    genres_choose = [k for k,v in x.items() if v]

    df_film = df_film[df_film.genres.isin(genres_choose)]
    df_film = df_film.drop_duplicates(subset=['movie'])
    df_film = df_film[df_film.budget >= 10000000]

    film_list = df_film.movie.unique()
    film_list.sort()
    film_data = c2.selectbox('Select film', film_list)

    df_mask = df_film[df_film.movie == film_data].reset_index()
    genres_string = df[df.movie == df_mask.movie[0]]['genres']

    # c1, c2 = st.columns((3, 4))
    try:
        c2.image(get_film_image(df_mask.poster_path.unique()[0]))
        c3.text("")
        c3.text("")
        c3.text("")
        c3.text("")
        c3.text("")
        c3.header(df_mask.overview[0])
        c3.write(f'Released : {df_mask.release_year[0]}')
        c3.write(f'Genres : {str(genres_string.values[0]).replace("[", "").replace("]", "")}')
        c3.write(f'Budget : {str(df_mask.budget.values[0])}')
        c3.write(f'Revenue : {str(df_mask.revenue.values[0])}')
    except:
        pass


    return