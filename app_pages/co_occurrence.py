import numpy as np
import streamlit as st
from app_dependencies.co_occurrence import get_co_occurrence_frame, get_edges_popularity, get_graph,\
                        get_no_edges_popularity
from app_dependencies.visuals import plotly_streamlit_texts, plotly_streamlit_layout
from app_dependencies.actor_deepdive import get_film_image
import plotly.express as px


def write(df, df_cast, df_actor_all):


    st.title('Actor Co-Occurence')
    net = get_graph(df, df_actor_all, year_threshold=None)

    df_edges = get_edges_popularity(net, df_cast)
    df_edges = df_edges.sort_values(by=['source'], ascending=False)

    fig = px.scatter(df_edges, x='target', y='source', size='edge_frequency', hover_name='node_plot',
                     color='edge_frequency', color_continuous_scale='YlGnBu')

    st.plotly_chart(plotly_streamlit_texts(plotly_streamlit_layout(fig, height=1200, width=1200), x_title=None,
                                           y_title=None))
    st.header('Films starred in together')
    c1, c2 = st.columns((1, 1))
    actors = list(df_actor_all.actor.unique())
    actors.sort()
    actor_1 = c1.selectbox('Actor 1:', actors)
    ac_1_mask = df_actor_all.actor == actor_1

    actors_co_star = df_actor_all[df_actor_all.movie.isin(df_actor_all[ac_1_mask].movie.unique())]['actor'].unique()
    actors_co_star.sort()
    actor_2 = c2.selectbox('Actor 2:', actors_co_star)

    ac_2_mask = df_actor_all.actor == actor_2

    results = np.intersect1d(df_actor_all[ac_1_mask].movie.unique(), df_actor_all[ac_2_mask].movie.unique())

    poster_urls = df_actor_all[df_actor_all.movie.isin(results)].drop_duplicates(subset=['movie']).sort_values(
        by='release_date')
    poster_urls = poster_urls['poster_path'].values

    width_needed = len(poster_urls)
    st.write(width_needed)
    if width_needed == 1:
        st.image(get_film_image(poster_urls[0]), width=700)

    elif width_needed == 2:
        c1, c2 = st.columns((5, 5))
        containers = [c1, c2]
        for i in range(len(poster_urls)):
            containers[i].image(get_film_image(poster_urls[i]), width=550)

    elif width_needed == 3:
        c1, c2, c3 = st.columns((4, 4, 4))
        containers = [c1, c2, c3]
        for i in range(len(poster_urls)):
            containers[i].image(get_film_image(poster_urls[i]), width=450)

    elif width_needed == 4:
        c1, c2, c3, c4 = st.columns((3, 3, 3, 3))
        containers = [c1, c2, c3, c4]
        for i in range(len(poster_urls)):
            containers[i].image(get_film_image(poster_urls[i]), width=385)

    else:

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns((2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1))
        containers = [c1, c2, c3, c4, c5, c6, c7, c8]
        for i in range(len(poster_urls)):
            containers[i].image(get_film_image(poster_urls[i]), width=190)

    return