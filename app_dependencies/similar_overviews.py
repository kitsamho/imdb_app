import pandas as pd
import numpy as np
import streamlit as st
import tensorflow_hub as hub

# find similar overviews

@st.cache
def get_encoder():
    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
    return use


@st.cache
def get_full_embeddings():
    dfs = []
    parts = ['3000', '6000', '9000', '12000']
    for i in parts:
        df = pd.read_csv(f'./data/overview_embeddings/embed_full_{i}.csv')
        dfs.append(df)
    return pd.concat(dfs).set_index('tmdb_id').drop_duplicates()


def get_vector_from_text(user_text, model_instance):
    vectorised_text = model_instance.fit_transform([user_text], reduce=False)
    vectorised_text_array = np.array(vectorised_text.T.iloc[1:][0])
    return vectorised_text_array


def mask_main_frame(df_main, threshold=10):
    return df_main[df_main.popularity >= threshold].set_index('tmdb_id')


def get_masked_embeddings(df_main, embed_raw, threshold=10):
    df_use = mask_main_frame(df_main, threshold=threshold)
    embed_raw_mask = embed_raw.reindex(index=df_use.index).dropna()
    return embed_raw_mask


def matrix_operation(embed_raw, text_vec):
    results = [(embed_raw.index[i], np.dot(text_vec, embed_raw.iloc[i])) for i in range(embed_raw.shape[0])]
    return pd.DataFrame(results)


def get_meta_data(df_main, df_matrix):
    df_use = df_main[['tmdb_id', 'movie', 'budget', 'poster_path', 'overview']]
    # df_use = df_use[df_use.budget >= df_use.budget.median()]
    df_final = pd.merge(df_use, df_matrix, left_on='tmdb_id', right_on=0, how='left').dropna().sort_values(by=1,
                                                                                                           ascending=False)
    return df_final