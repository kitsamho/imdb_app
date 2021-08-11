import pandas as pd
import os
import json
import streamlit as st
import plotly.express as px
from app_dependencies.semantics import get_embeddings, getAllNounChunks, visualise_nounchunks
from app_dependencies.visuals import plotly_streamlit_layout, plotly_streamlit_texts
from app_dependencies.actor_deepdive import get_film_image

def write(df):
    st.title('Semantics')

    df = df[['tmdb_id', 'movie', 'overview', 'genres', 'genres_new', 'poster_path',
                       'popularity', 'release_year', 'vote_average', 'budget', 'revenue']]

    st.header('Exploring Film Descriptions')

    film_list = df.movie.unique()
    film_list.sort()
    film_data = st.selectbox('Select film', film_list)

    df_mask = df[df.movie == film_data].reset_index()
    c1, c2 = st.columns((3, 4))
    c1.image(get_film_image(df_mask.poster_path.unique()[0]))
    c2.subheader(df_mask.overview[0])
    c2.write(f'Released : {df_mask.release_year[0]}')
    c2.write(f'Genres : {str(df_mask.genres[0]).replace("[", "").replace("]", "")}')

    use_paths = ['./data/overview_embeddings/overview_embeddings_6000.csv',
                 './data/overview_embeddings/overview_embeddings_12126.csv']

    bert_paths = ['./data/overview_embeddings/overview_embeddings_bert6000.csv',
                 './data/overview_embeddings/overview_embeddings_bert12126.csv']

    genres_choose = ['comedy', 'action', 'drama', 'scifi', 'horror', 'crime', ]

    st.header('Visualising Film Descriptions Using Text Representations')

    reps = st.selectbox('Choose BERT or Universal Sentence Encoding', ('BERT', 'Universal Sentence Encoding'))

    if reps == 'BERT':
        rep_path = bert_paths
    else:
        rep_path = use_paths

    df_embed = get_embeddings(rep_path)

    df_semantic_plot = pd.merge(get_embeddings(use_paths), df, how='left').dropna(subset=['genres_new'])
    df_semantic_plot = df_semantic_plot[df_semantic_plot.genres_new.isin(genres_choose)]
    df_semantic_plot = df_semantic_plot.dropna()
    df_semantic_plot.genres_new = df_semantic_plot.genres_new.apply(lambda x: x.capitalize())

    df_semantic_plot.overview = df_semantic_plot.overview.str.wrap(30)
    df_semantic_plot.overview = df_semantic_plot.overview.apply(lambda x: x.replace('\n', '<br>'))
    df_semantic_plot['plot_data'] = df_semantic_plot.movie + '<br><br>' + df_semantic_plot.overview

    c1, c2 = st.columns((2, 2))
    dimensions = c1.selectbox('Components', ('2', '3'), index=0)
    size = c2.selectbox('Size by', ('Popularity', 'Budget', 'Revenue', 'Vote Average', 'No Sizing'), index=4)

    size_dic = {'Vote Average': 'vote_average', 'Popularity': 'popularity',
                'Budget': 'budget', 'Revenue': 'revenue', 'No Sizing': None}

    if dimensions == '2':

        fig = px.scatter(df_semantic_plot[df_semantic_plot.popularity >= 5], x='ts_2_x', y='ts_2_y',
                         color='genres_new', opacity=0.8, size=size_dic[size], hover_name='plot_data')

    else:
        fig = px.scatter_3d(df_semantic_plot[df_semantic_plot.popularity >= 5], x='ts_3_x', y='ts_3_y', z='ts_3_z',
                            color='genres_new', opacity=0.8, size=size_dic[size], hover_name='plot_data')

    st.plotly_chart(plotly_streamlit_layout(fig, height=1000, width=1000))

    st.header('Part of Speech Analysis')

    genre_analyse = st.selectbox('Choose genre', ('Action', 'Comedy', 'Drama', 'SciFi', 'Horror', 'Crime'),
                                 index=4).lower()
    st.subheader(f'Most common nouns used in {genre_analyse.capitalize()}')

    options_dic = {'action': 'Oranges',
                   'comedy': 'Blues',
                   'drama': 'Greens',
                   'scifi': 'Reds',
                   'horror': 'Purples',
                   'crime': 'Blues'}

    path = os.getcwd() + f'/data/nouns/{genre_analyse}.txt'

    with open(path) as f:
        x = pd.DataFrame(json.loads(f.read()))

    df_nouns = getAllNounChunks(x, 'spaCy_nouns', 'spaCy_noun_chunk', chunk_token=2, top_nouns=60)

    fig = visualise_nounchunks(df_nouns, colour_use=options_dic[genre_analyse])

    st.plotly_chart(plotly_streamlit_layout(fig, width=1500, height=800))

    return