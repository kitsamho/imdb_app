import pandas as pd
import os
import json
import streamlit as st
import plotly.express as px
from app_dependencies.semantics import get_embeddings, getAllNounChunks, visualise_nounchunks
from app_dependencies.visuals import plotly_streamlit_layout, plotly_streamlit_texts
from app_dependencies.film_deepdive import get_film_image, get_actor_image




def write(df, df_actor_all):
    st.title('Film Deepdive')

    # st.write(df_actor_all.head())

    df = df[['tmdb_id', 'movie', 'overview', 'genres', 'genres_new', 'poster_path',
                       'popularity', 'release_year', 'vote_average', 'budget', 'revenue']]

    df = df[df.movie.isin(df_actor_all.movie.unique())]
    df_film = df.explode('genres')
    df_film = df_film.dropna(subset=['genres'])
    st.text("")
    st.text("")
    st.write('Select genres to include:')


    x = {genre:st.checkbox(genre) for genre in df_film.genres.unique()}

    genres_choose = [k for k,v in x.items() if v]

    df_film = df_film[df_film.genres.isin(genres_choose)]
    df_film = df_film.drop_duplicates(subset=['movie'])
    df_film = df_film[df_film.budget >= 10000000]

    film_list = df_film.movie.unique()
    film_list.sort()
    film_data = st.selectbox('Select film', film_list)

    df_mask = df_film[df_film.movie == film_data].reset_index()
    genres_string = df[df.movie == df_mask.movie[0]]['genres']


    try:
        c1,c2 = st.columns((2,2))
        c1.image(get_film_image(df_mask.poster_path.unique()[0]))


        c2.header(df_mask.overview[0])
        c2.write(f'Released : {df_mask.release_year[0]}')
        c2.write(f'Genres : {str(genres_string.values[0]).replace("[", "").replace("]", "")}')
        c2.write(f'Budget : {str(df_mask.budget.values[0])}')
        c2.write(f'Revenue : {str(df_mask.revenue.values[0])}')
    except:
        pass
    x = df_actor_all[df_actor_all.movie == film_data][['actor','character','profile_path']].head(10).reset_index()
    x['string'] = x['actor'] + ' ('+x['character']+')'

    st.header(f'Main Cast of {film_data}')
    st.write(df_actor_all.head())
    actor_images = [get_actor_image(i) for i in x.profile_path]
    c1, c2, c3, c4, c5= st.columns((1,1,1,1,1))
    c1.image(actor_images[0],width=150)
    c1.text(x['actor'][0])
    c2.image(actor_images[1], width=150)
    c2.text(x['actor'][1])
    c3.image(actor_images[2], width=150)
    c3.text(x['actor'][2])
    c4.image(actor_images[3], width=150)
    c4.text(x['actor'][3])
    c5.image(actor_images[4], width=150)
    c5.text(x['actor'][4])
    c6, c7, c8, c9, c10 = st.columns((1, 1, 1, 1, 1))
    c6.image(actor_images[5], width=150)
    c6.text(x['actor'][5])
    c7.image(actor_images[6], width=150)
    c7.text(x['actor'][6])
    c8.image(actor_images[7], width=150)
    c8.text(x['actor'][7])
    c9.image(actor_images[8], width=150)
    c9.text(x['actor'][8])
    c10.image(actor_images[9], width=150)
    c10.text(x['actor'][9])


    return