import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from app_dependencies.actor_deepdive import get_actor_frame, get_data_from_dic, get_actor_image_url, get_film_image
from app_dependencies.visuals import plotly_streamlit_texts, plotly_streamlit_layout

streamlit_font_path = ('./assets/fonts/IBMPlexSans-Regular.ttf')
light = 'rgb(131,199,161)'
medium = 'rgb(6,180,226)'
dark = 'rgb(3,37,65)'


def write(df_actor_all):
    st.title('Actor Deepdive')

    df_actor_all = df_actor_all[df_actor_all.popularity_actor >= 5]
    actor_list = df_actor_all.actor.unique()
    actor_list.sort()
    actor_data = st.selectbox('Select which actor to analyse', actor_list, index=9)
    df_actor_specific_frame = get_actor_frame(df_actor_all, actor_data)

    c1, mid, c2 = st.columns((1, 0.8, 2))
    actor_image = get_actor_image_url(df_actor_specific_frame)[0]

    actor_image_resize = actor_image.resize((1, 1))

    actor = df_actor_specific_frame.actor.unique()[0]
    c1.header(actor)
    c1.image(actor_image, width=430)

    c2.header('Genres Known for')
    genres_in = pd.DataFrame(df_actor_specific_frame.explode('genres')['genres'].value_counts())
    genres_in_pie = go.Figure(data=[go.Pie(labels=genres_in.index, values=genres_in['genres'], textinfo='label+percent',
                                           hole=0.3, )])
    genres_in_pie.update_layout(width=600, height=600, showlegend=False)
    genres_in_pie.update_traces(marker=dict(colors=px.colors.sequential.YlGnBu))
    genres_in_pie.update_traces(textfont_size=16)
    genres_in_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    c2.plotly_chart(genres_in_pie)

    st.header('Filmography')
    df_filmography = df_actor_specific_frame.copy()
    df_filmography['release_date'] = df_filmography['release_date'].apply(lambda x: str(x).replace('00:00:00', ''))
    df_filmography['character_year'] = df_filmography['character'] + ' | ' + df_filmography['release_date']
    order = df_filmography.sort_values(by='release_date')['movie'].values

    film_fig = plotly_streamlit_layout(px.bar(df_filmography, x='movie', y='vote_average', orientation='v', \
                                              color_discrete_map={'movie': 'black'}, category_orders={'movie': order},
                                              text='character_year'), \
                                       width=1000, height=775)
    film_fig.update_traces(marker_color=light)
    st.plotly_chart(plotly_streamlit_texts(film_fig, x_title=None, y_title=None))

    st.header('Word Cloud of Keywords')
    texts = get_data_from_dic(df_actor_specific_frame, 'keywords', 'name', string_return=True)

    image_mask = np.array(actor_image)
    wordcloud = WordCloud(font_path=streamlit_font_path, background_color="white", contour_color='black',
                          prefer_horizontal=True, width=1100, height=400,
                          mode="RGBA", max_words=2000).generate_from_text(texts)

    plt.figure(figsize=(20, 10), dpi=2400)
    plt.imshow(wordcloud)
    plt.axis("off")
    fig1 = plt.gcf()

    st.pyplot(fig1)

    st.header('Interactive Analysis')
    my_expander = st.expander(label='Click for info')
    my_expander.write('Feel free to experiment with different variables at different data points!')

    c1, c2 = st.columns((2, 2))
    x_ax = c1.selectbox('X Axis', ('Revenue', 'Budget', 'Release Date', 'Film Popularity', 'Review Average'), index=2)
    y_ax = c1.selectbox('Y Axis', ('Revenue', 'Budget', 'Release Date', 'Film Popularity', 'Review Average'), index=0)
    colour = c2.selectbox('Colour by', ('Revenue', 'Budget', 'Release Date', 'Film Popularity', 'Review Average'),
                          index=4)
    size = c2.selectbox('Size by', ('Revenue', 'Budget', 'Release Date', 'Film Popularity', 'Review Average'), index=1)

    film_dic = {'Revenue': 'revenue',
                'Budget': 'budget',
                'Release Date': 'release_date',
                'Film Popularity': 'popularity',
                "Review Average": 'vote_average'}

    film_time = px.scatter(df_actor_specific_frame, x=film_dic[x_ax], opacity=0.7, size=film_dic[size]
                           , y=film_dic[y_ax], color=film_dic[colour], hover_name='movie',
                           color_continuous_scale='ylgnbu')

    st.plotly_chart(plotly_streamlit_texts(plotly_streamlit_layout(film_time, height=650, width=1000),
                                           x_title=film_dic[x_ax], y_title=film_dic[y_ax]))


    return