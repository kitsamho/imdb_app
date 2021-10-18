import pandas as pd
import numpy as np
import streamlit as st
from app_dependencies.semantics import get_embeddings
from app_dependencies.film_deepdive import get_film_image, get_actor_image
from app_dependencies.actor_deepdive import get_data_from_dic


def get_details(df, col, key, string_return):
    results = get_data_from_dic(df, col, key, string_return=string_return)
    results_list = [results[i] for i in range(len(results))]
    return results_list


def write(df, df_actor_all):
    st.title('Film Deepdive')

    df_details = df_actor_all.drop_duplicates(subset=['movie'])

    df = df[['tmdb_id', 'movie', 'overview', 'genres', 'genres_new', 'poster_path',
             'popularity', 'release_year', 'vote_average', 'budget', 'revenue']]

    df = df[df.movie.isin(df_actor_all.movie.unique())]
    df_film = df.explode('genres')
    df_film = df_film.dropna(subset=['genres'])

    genres_choose = st.multiselect('Select genres to include:',df_film.genres.unique())

    try:

        df_film = df_film[df_film.genres.isin(genres_choose)]
        df_film = df_film.drop_duplicates(subset=['movie'])
        df_film = df_film[df_film.budget >= 10000000]

        film_list = df_film.movie.unique()
        film_list.sort()

        film_data = st.selectbox('Select film', film_list)
        c1, c2 = st.columns((2, 4))

        df_mask = df_film[df_film.movie == film_data].reset_index()
        genres_string = df[df.movie == df_mask.movie[0]]['genres']

        df_details_film = df_details[df_details.movie == film_data].reset_index()
        production_companies = get_details(df_details_film, 'production_companies', 'name', False)
        production_countries = get_details(df_details_film, 'production_countries', 'name', False)

        try:

            c1.image(get_film_image(df_mask.poster_path.unique()[0]))

            # c2.text("")
            # c2.text("")
            # c2.text("")
            # c2.text("")
            # c2.text("")
            c2.header(df_mask.overview[0])
            c2.write(f'Released : {df_mask.release_year[0]}')
            c2.write(f'Genres : {str(genres_string.values[0]).replace("[", "").replace("]", "")}')
            c2.write(f'Budget : {str(df_mask.budget.values[0])}')
            c2.write(f'Revenue : {str(df_mask.revenue.values[0])}')
            c2.write(f'Profit : ')
            c2.write(f'Production Companies :{str(production_companies).replace("[", "").replace("]", "")}')
            c2.write(f'Production Countries :{str(production_countries).replace("[", "").replace("]", "")}')
        except:
            pass

        st.header('Main Cast')
        x = df_actor_all[df_actor_all.movie == film_data][['actor', 'character', 'profile_path']].head(10).reset_index()
        x['string'] = x['actor'] + ' (' + x['character'] + ')'

        actor_images = [get_actor_image(i) for i in x.profile_path]
        c1, c2, c3, c4, c5 = st.columns((1, 1, 1, 1, 1))

        first_stack = [c1, c2, c3, c4, c5]
        for i in range(len(first_stack)):
            try:
                first_stack[i].image(actor_images[i], width=150)
                first_stack[i].text(x['actor'][i])
            except:
                pass
        c6, c7, c8, c9, c10 = st.columns((1, 1, 1, 1, 1))

        second_stack = [c6, c7, c8, c9, c10]
        for i in range(len(second_stack)):
            try:
                second_stack[i].image(actor_images[i + 5], width=150)
                second_stack[i].text(x['actor'][i + 5])
            except:
                pass

        st.header('Reviews')
        reviews = get_details(df_details_film, 'reviews', 'content', False)
        [st.write(i) for i in reviews]

        st.title('You might also like')
        use_paths = ['./data/overview_embeddings/embed_full_3000.csv',
                     './data/overview_embeddings/embed_full_6000.csv',
                     './data/overview_embeddings/embed_full_9000.csv',
                     './data/overview_embeddings/embed_full_12000.csv']

        df_all_embeds = pd.merge(get_embeddings(use_paths, drop_full_embeddings=False), df, how='left', left_index=True, \
                                 right_on='tmdb_id').dropna(subset=['overview']).drop_duplicates(subset=['movie'])

        film_embed = df_all_embeds[df_all_embeds.movie == film_data].iloc[:, :512]  # embeddings for the film selected
        embed_df = pd.DataFrame(df_all_embeds.iloc[:, :512]).T  # embeddings for all films

        # get the dot product of the film embedding and all films
        similarity_df = pd.DataFrame(np.dot(np.array(film_embed), embed_df)).T.set_index(df_all_embeds.movie)

        similarity_df = pd.merge(similarity_df, df[['movie', 'poster_path']].drop_duplicates(subset=['movie']) \
                                 , how='left', left_index=True, right_on='movie')

        similarity_df = similarity_df.sort_values(by=0, ascending=False).reset_index(drop=True)

        poster_images = [get_film_image(i) for i in similarity_df.poster_path[1:12]]
        c1, c2, c3, c4, c5 = st.columns((1, 1, 1, 1, 1))

        first_stack = [c1, c2, c3, c4, c5]
        for i in range(len(first_stack)):
            try:
                first_stack[i].image(poster_images[i], width=150)
                first_stack[i].write(round(similarity_df[0][i + 1], 2))
            except:
                pass
        c6, c7, c8, c9, c10 = st.columns((1, 1, 1, 1, 1))

        second_stack = [c6, c7, c8, c9, c10]
        for i in range(len(second_stack)):
            try:
                second_stack[i].image(poster_images[i + 5], width=150)
                second_stack[i].write(round(similarity_df[0][i + 6], 2))
            except:
                pass
    except:
        pass
    return
