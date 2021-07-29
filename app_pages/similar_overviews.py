import pandas as pd
import numpy as np
import streamlit as st
import tensorflow_hub as hub

def write():
    embed_raw = get_full_embeddings()
    use = get_encoder()
    # encoder = copy.deepcopy(use)
    user_text = st.text_area('Write your film description here and AI will find you something similar')
    text_vec = np.array(use([user_text])[0])
    # text_vec = get_vector_from_text(user_text, use)
    # st.write(text_vec)

    embed_raw_mask = get_masked_embeddings(df, embed_raw)
    df_matrix = matrix_operation(embed_raw_mask, text_vec)
    results = get_meta_data(df, df_matrix).head(10)
    poster_paths = results['poster_path'].to_list()
    overviews = results['overview'].to_list()
    similarity = results[1].to_list()

    c1, c2, c3 = st.beta_columns((4, 4, 4))
    containers = [c1, c2, c3]
    for i in range(len(poster_paths)):
        containers[i].header(f'Similarity Score:{similarity[i]}')
        containers[i].image(get_film_image(poster_paths[i]), width=450)
        containers[i].subheader(overviews[i])

    return
