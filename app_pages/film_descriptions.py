import pandas as pd
import os
import json
import streamlit as st
import plotly.express as px
from app_dependencies.semantics import get_embeddings, getAllNounChunks, visualise_nounchunks
from app_dependencies.visuals import plotly_streamlit_layout, plotly_streamlit_texts
from app_dependencies.actor_deepdive import get_film_image


def write(df):

    use_paths = ['./data/overview_embeddings/overview_embeddings_6000.csv',
                 './data/overview_embeddings/overview_embeddings_12126.csv']

    bert_paths = ['./data/overview_embeddings/overview_embeddings_bert6000.csv',
                 './data/overview_embeddings/overview_embeddings_bert12126.csv']

    genres_choose = ['comedy', 'action', 'drama', 'scifi', 'horror', 'crime', ]
    st.title('Film Descriptions')

    st.header('Text Embeddings')

    reps = st.selectbox('Choose BERT or Universal Sentence Encoding', ('BERT', 'Universal Sentence Encoding'))

    if reps == 'BERT':
        rep_path = bert_paths
    else:
        rep_path = use_paths

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

    st.plotly_chart(plotly_streamlit_layout(fig, height=850, width=850))

    st.header('Part of Speech Analysis (Noun Phrases)')

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

    st.plotly_chart(plotly_streamlit_layout(fig, width=1100, height=800))

    return