import pandas as pd
from samutil import NetworkTransformer
import streamlit as st


# co-occurrence page
def get_co_occurrence_frame(df, df_actor, pop_mask=None, genre_mask=[], actor_mask=[]):
    if pop_mask:
        df_actor = actor_popularity_mask(df_actor, pop_mask=pop_mask)

    actor_grouped = pd.DataFrame(df_actor.groupby('movie')['actor'].agg(lambda x: x.to_list()))
    cols_use = ['movie', 'popularity', 'release_date', 'release_year', 'vote_average', 'budget', 'revenue', 'genres']
    df = df[cols_use].set_index('movie')
    df_co = pd.merge(df, actor_grouped, left_index=True, right_index=True, how='right').reset_index()

    if genre_mask:
        films_in_genre = film_mask(df_co, col='genres', mask=genre_mask)
        df_co = df_co[df_co.movie.isin(films_in_genre)]

    if actor_mask:
        films_in_genre = film_mask(df_co, col='actor', mask=actor_mask)
        df_co = df_co[df_co.movie.isin(films_in_genre)]

    return df_co


def get_no_edges_popularity(graph):
    df = pd.DataFrame(graph.no_edge_df.groupby('source')['target'].agg(lambda x: x.to_list()))
    df['no_edge_count'] = df['target'].apply(lambda x: len(x))
    return df


def get_edges_popularity(graph, cast):
    first_merge = pd.merge(graph.edge_df, cast[['actor', 'popularity_actor']], left_on='source', right_on='actor',
                           how='left')
    second_merge = pd.merge(first_merge, cast[['actor', 'popularity_actor']], left_on='target', right_on='actor',
                            how='left')
    second_merge = second_merge.drop(columns=['actor_x', 'actor_y'])
    second_merge = second_merge.rename(
        columns={'popularity_actor_x': 'source_popularity', 'popularity_actor_y': 'target_popularity'})
    second_merge = second_merge.drop_duplicates(subset=['source', 'target'])
    second_merge['node_plot'] = second_merge['source'] + ' & ' + second_merge['target']
    return second_merge
# creates a graph object

@st.cache
def get_graph(df, df_actor, actor_mask=None, year_threshold=None):
    df = get_co_occurrence_frame(df, df_actor, actor_mask=actor_mask, pop_mask=10)
    if year_threshold:
        df = df[df.release_year >= year_threshold]
    Net = NetworkTransformer(df['actor'].dropna())
    Net.fit_transform()
    return Net


def actor_popularity_mask(df, pop_mask=5):
    return df[df.popularity_actor >= pop_mask]


def film_mask(df, col, mask):
    df = df.explode(col)
    df = df[df[col].isin(mask)]
    return df.movie