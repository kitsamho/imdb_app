import os

import pandas as pd
import numpy as np
import json

import streamlit as st

import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import spacy

model = spacy.load('en_core_web_sm')




st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
streamlit_font_path = ('./assets/fonts/IBMPlexSans-Regular.ttf')
st.sidebar.image('./assets/tmdb_logo_s.png')






def get_cast_frame(path):
    with open(path) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    df = df.rename(columns={'popularity': 'popularity_actor', 'name': 'actor'})
    return df



@st.cache
def get_data_frames():
    df = pd.read_json(os.getcwd() + '/data/main/tmdb_data_main.json')
    df.release_date = pd.to_datetime(df.release_date)
    cast = get_cast_frame(os.getcwd() + '/data/cast_crew/cast_3.jsonl')
    cast = cast[cast.adult == False]
    df_cast = cast[['tmdb_id', 'actor', 'popularity_actor', 'profile_path', 'character']]
    df_actor_all = pd.merge(cast, df, left_on='tmdb_id', right_on='tmdb_id', how='right').dropna()
    return df, df_cast, df_actor_all

df, df_cast, df_actor_all = get_data_frames()






navigation_buttons = {
                        "About": about,
                        "Overview": cross_section_analysis,
                        "Actor Deepdive": time_series_analysis,
                        "Semantics": time_series_analysis,
                        "Find Similar Descriptions": time_series_analysis,

}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(navigation_buttons.keys()))
if selection == 'About':
    df = df_code_book
else:
    df = df_final
page = navigation_buttons[selection]
page.write(df)











if active_tab == "Home":
    st.write('ed')

elif active_tab == "Exploratory Analysis":
    time_response = st.selectbox('Do you want analyse a period or a specific year?',
                                 ('A period', 'A specific decade', 'A specific year'))

    if time_response == 'A period':
        year = st.slider('Select earliest date', df.release_date.min().year, \
                         df.release_date.max().year, df.release_date.min().year)
        df = df[df['release_date'] >= str(year)]
        year_string = str(year) + ' to ' + str(df.release_date.max().year)

    elif time_response == 'A specific year':
        years = list(set([i.year for i in df.release_date]))[1:]
        years.sort(reverse=True)
        year = st.selectbox('Select specific year:', years)
        df = df[(df.release_date > str(year)) & (df.release_date < str(year + 1))]
        year_string = year

    else:
        decades = st.selectbox('Select specific decade:', ('60s', '70s', '80s', '90s', '00s', '10s', '20s'))
        decade_dic = {'60s': (1960, 1970),
                      '70s': (1970, 1980),
                      '80s': (1980, 1990),
                      '90s': (1990, 2000),
                      '00s': (2000, 2010),
                      '10s': (2010, 2020),
                      '20s': (2020, 2022)}
        df = df[(df.release_date > str(decade_dic[decades][0])) & (df.release_date < str(decade_dic[decades][1]))]
        year_string = 'in the ' + decades

    st.subheader(f'Budget, Revenue and Profit For Films {year_string}')
    df['budget_revenue'] = df['budget'] + df['revenue']
    df = df.sort_values(by='budget_revenue', ascending=False)
    if df.shape[0] > 100:
        df = df.head(100)

    fig_revenue = px.bar(df, x='movie', y=['budget', 'revenue'], orientation='v',
                         color_discrete_map={'budget': medium, 'revenue': light})

    data_type = st.radio('Absolutes or normalised', ('Absolutes', 'Normalised'))

    if data_type == 'Absolutes':
        barnorm = None
    else:
        barnorm = 'percent'
    st.plotly_chart(plotly_streamlit_layout(plotly_streamlit_texts(fig_revenue, \
                                                                   x_title='Movie', y_title='Dollars'), \
                                            barmode='stack', barnorm=barnorm, width=1600, height=650))

    try:
        st.subheader(f'Genres {year_string}')

        genre_period = df.explode('genres').dropna(subset=['genres'])
        genre_period = genre_period[genre_period.vote_average > 0]
        genre_period_fig = px.treemap(genre_period, path=[px.Constant("Films"), 'genres', 'movie'],
                                      values='revenue',
                                      color='vote_average',
                                      color_continuous_scale='YlGnBu')
        genre_period_fig.update_layout(width=1600, height=650,
                                       margin=dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(genre_period_fig)

    except:
        pass

    st.subheader(f'Busiest Actors {year_string} : Film Counts')

    actor_value_counts = df_actor_all.actor.value_counts()

    actor_films = df_actor_all.groupby('actor')['movie'].agg(lambda x: x.to_list())
    actor_merge = pd.merge(actor_value_counts, actor_films, left_index=True, right_index=True,
                           how='inner').reset_index()
    if actor_merge.shape[0] > 50:
        actor_merge = actor_merge.head(50)

    actor_merge.columns = ['actor', 'count', 'films']

    busiest_actor_fig = px.bar(actor_merge, x=actor_merge['actor'], y=actor_merge['count'])
    busiest_actor_fig.update_traces(marker_color=light)

    st.plotly_chart(
        plotly_streamlit_layout(plotly_streamlit_texts(busiest_actor_fig, x_title=None, y_title=None), width=1600,
                                height=650))

    st.subheader(f'Busiest Actors {year_string} : Film Titles')
    c1, c2, c3 = st.beta_columns((0.5, 2, 0.4))
    actor_merge = actor_merge.explode('films')
    actor_merge = pd.merge(actor_merge, df[['budget', 'movie', 'popularity', 'vote_average']], how='left',
                           left_on='films', right_on='movie')
    busiest_actor_fig_2 = px.sunburst(actor_merge, path=['actor', 'films'], values='budget', color='vote_average',
                                      color_continuous_scale= \
                                          px.colors.sequential.YlGnBu)

    busiest_actor_fig_2.update_layout(width=1000, height=1000, showlegend=False)
    # busiest_actor_fig_2.update_traces(marker=dict(colors=px.colors.qualitative.Pastel2))
    busiest_actor_fig_2.update_traces(textfont_size=12, marker=dict(line=dict(color='#000000', width=0.5)))
    busiest_actor_fig_2.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    c2.plotly_chart(busiest_actor_fig_2)

elif active_tab == "Actor Filmography":

    actor_list = df_actor_all.actor.unique()
    actor_list.sort()
    actor_data = st.selectbox('Select which actor to analyse', actor_list, index=0)
    df_actor_specific_frame = get_actor_frame(df_actor_all, actor_data)

    c1, mid, c2 = st.beta_columns((1, 0.2, 2))
    actor_image, actor_url = get_actor_image_url(df_actor_specific_frame)

    actor_image_resize = actor_image.resize((1, 1))

    actor = df_actor_specific_frame.actor.unique()[0]
    c1.header(actor)
    c1.markdown(f"![]({actor_url})")

    c2.header('Genres Known for')
    genres_in = pd.DataFrame(df_actor_specific_frame.explode('genres')['genres'].value_counts())
    genres_in_pie = go.Figure(data=[go.Pie(labels=genres_in.index, values=genres_in['genres'], textinfo='label+percent',
                                           hole=0.3, )])
    genres_in_pie.update_layout(width=700, height=700, showlegend=False)
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
                                       width=1600, height=775)
    film_fig.update_traces(marker_color=light)
    st.plotly_chart(plotly_streamlit_texts(film_fig, x_title=None, y_title=None))

    st.header('Word Cloud of Keywords')
    texts = get_data_from_dic(df_actor_specific_frame, 'keywords', 'name', string_return=True)

    image_mask = np.array(actor_image)
    wordcloud = WordCloud(font_path=streamlit_font_path, background_color="white", contour_color='black',
                          prefer_horizontal=True, width=1600, height=400,
                          mode="RGBA", max_words=2000).generate_from_text(texts)

    plt.figure(figsize=(20, 10), dpi=2400)
    plt.imshow(wordcloud)
    plt.axis("off")
    fig1 = plt.gcf()

    st.pyplot(fig1)

    st.header('Interactive Analysis')
    my_expander = st.beta_expander(label='Click for info')
    c1, c2 = st.beta_columns((2, 2))
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


    st.plotly_chart(plotly_streamlit_texts(plotly_streamlit_layout(film_time, height=650, width=1400),
                                           x_title=film_dic[x_ax], y_title=film_dic[y_ax]))

    st.header('Reviews')

    film_to_review = st.selectbox('Choose Film', df_actor_specific_frame.movie.unique())

    df_review = df_actor_specific_frame[df_actor_specific_frame.movie == film_to_review]

    reviews = get_data_from_dic(df_review, 'reviews', 'content', string_return=False)
    for i in range(len(reviews)):
        st.write(reviews[i])

elif active_tab == "Actor Co-Occurrence":
    st.header('Actor Co-Occurence')
    net = get_graph(df, df_actor_all, year_threshold=None)

    df_edges = get_edges_popularity(net, df_cast)
    df_edges = df_edges.sort_values(by=['source'], ascending=False)

    fig = px.scatter(df_edges, x='target', y='source', size='edge_frequency', hover_name='node_plot',
                     color='edge_frequency', color_continuous_scale='YlGnBu')

    st.plotly_chart(plotly_streamlit_texts(plotly_streamlit_layout(fig, height=1600, width=1600), x_title=None,
                                           y_title=None))
    st.header('Films starred in together')
    c1, c2 = st.beta_columns((1, 1))
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
        c1, c2 = st.beta_columns((5, 5))
        containers = [c1, c2]
        for i in range(len(poster_urls)):
            containers[i].image(get_film_image(poster_urls[i]), width=550)

    elif width_needed == 3:
        c1, c2, c3 = st.beta_columns((4, 4, 4))
        containers = [c1, c2, c3]
        for i in range(len(poster_urls)):
            containers[i].image(get_film_image(poster_urls[i]), width=450)

    elif width_needed == 4:
        c1, c2, c3, c4 = st.beta_columns((3, 3, 3, 3))
        containers = [c1, c2, c3, c4]
        for i in range(len(poster_urls)):
            containers[i].image(get_film_image(poster_urls[i]), width=385)

    else:

        c1, c2, c3, c4, c5, c6, c7, c8 = st.beta_columns((2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1))
        containers = [c1, c2, c3, c4, c5, c6, c7, c8]
        for i in range(len(poster_urls)):
            containers[i].image(get_film_image(poster_urls[i]), width=190)

elif active_tab == "Semantics":

    df_main = pd.read_json('./data/main/tmdb_data_main.json')
    df_main = df_main[['tmdb_id', 'movie', 'overview', 'genres', 'genres_new', 'poster_path',
                       'popularity', 'release_year', 'vote_average', 'budget', 'revenue']]

    st.header('Exploring Film Descriptions')

    film_list = df_main.movie.unique()
    film_list.sort()
    film_data = st.selectbox('Select film', film_list)

    df_mask = df_main[df_main.movie == film_data].reset_index()
    c1, c2 = st.beta_columns((3, 4))
    c1.image(get_film_image(df_mask.poster_path.unique()[0]))
    c2.subheader(df_mask.overview[0])
    c2.write(f'Released : {df_mask.release_year[0]}')
    c2.write(f'Genres : {str(df_mask.genres[0]).replace("[", "").replace("]", "")}')

    use_paths = ['./data/overview_embeddings/overview_embeddings_6000.csv',
                 './data/overview_embeddings/overview_embeddings_12126.csv']

    genres_choose = ['comedy', 'action', 'drama', 'scifi', 'horror', 'crime', ]

    st.header('Visualising Film Descriptions Using Text Representations')

    # reps = st.selectbox('Choose BERT or Universal Sentence Encoding', ('BERT', 'Universal Sentence Encoding'))
    #
    # if reps == 'BERT':
    #     rep_path = bert_paths
    # else:
    #     rep_path = use_paths

    # df_embed = get_embeddings(rep_path)
    df_semantic_plot = pd.merge(get_embeddings(use_paths), df_main, how='left').dropna(subset=['genres_new'])
    df_semantic_plot = df_semantic_plot[df_semantic_plot.genres_new.isin(genres_choose)]
    df_semantic_plot = df_semantic_plot.dropna()
    df_semantic_plot.genres_new = df_semantic_plot.genres_new.apply(lambda x: x.capitalize())

    df_semantic_plot.overview = df_semantic_plot.overview.str.wrap(30)
    df_semantic_plot.overview = df_semantic_plot.overview.apply(lambda x: x.replace('\n', '<br>'))
    df_semantic_plot['plot_data'] = df_semantic_plot.movie + '<br><br>' + df_semantic_plot.overview

    c1, c2 = st.beta_columns((2, 2))
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


elif active_tab == 'Find Similar Descriptions':




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



else:
    st.error("Something has gone terribly wrong.")
