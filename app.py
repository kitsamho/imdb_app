import pandas as pd
import numpy as np
import json

import sys
import os


import streamlit as st
st.set_page_config(layout="wide")

import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from copy import copy, deepcopy
import networkx as nx

import spacy
model = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.getenv('SAM_PATH'))
from samutil import NetworkGeneration, most_common_tokens, UniversalSentenceEncoder
st.set_option('deprecation.showPyplotGlobalUse', False)


streamlit_font_path = ('./assets/fonts/IBMPlexSans-Regular.ttf')

def get_person_data_new(path):
    with open(path) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    df = df.rename(columns={'popularity': 'popularity_actor', 'name': 'actor'})
    return df


def get_actor_frame(df_actor_all, actor):
    return df_actor_all[df_actor_all.actor == actor]


def get_actor_image_url(df_actor_specific_frame):
    root_img_path = 'https://image.tmdb.org/t/p/w500'
    url = root_img_path + df_actor_specific_frame.profile_path.unique()[0]
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img, url

def get_poster(poster_url):
    response = requests.get(poster_url)
    img = Image.open(BytesIO(response.content))
    return img


def plotly_streamlit_layout(fig, barmode=None, barnorm=None, height=None,width=None):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      barmode=barmode,
                      barnorm=barnorm,
                      height = height,
                      width = width)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig.update_layout(margin=dict(l=50, r=50, b=50, t=50, pad=2))
    fig.update_layout(bargap=0.03)

    return fig


def plotly_streamlit_texts(fig, x_title, y_title):
    fig.update_layout(yaxis=dict(title=y_title, titlefont_size=10, tickfont_size=10),
                      xaxis=dict(title=x_title, titlefont_size=10, tickfont_size=10))

    return fig


def get_data_from_dic(df_actor_specific_frame, col, key, string_return=True):
    strings = []
    for movie in df_actor_specific_frame[col]:
        try:
            for keyword in movie:
                try:
                    strings.append(keyword[key])
                except:
                    pass
        except:
            pass

    if string_return:
        return ' '.join(strings)
    else:
        return [' '.join(i.split()) for i in strings]


def actor_popularity_mask(df,pop_mask=5):
    return df[df.popularity_actor >= pop_mask]


def movie_mask(df,col,mask):
    df = df.explode(col)
    df = df[df[col].isin(mask)]
    return df.movie


def get_co_occurrence_frame(df, df_actor, pop_mask=None, genre_mask=[], actor_mask=[]):
    if pop_mask:
        df_actor = actor_popularity_mask(df_actor, pop_mask=pop_mask)

    actor_grouped = pd.DataFrame(df_actor.groupby('movie')['actor'].agg(lambda x: x.to_list()))

    cols_use = ['movie', 'popularity', 'release_date', 'release_year', 'vote_average', 'budget', 'revenue', 'genres']
    df = df[cols_use].set_index('movie')

    df_co = pd.merge(df, actor_grouped, left_index=True, right_index=True, how='right').reset_index()

    if genre_mask:
        films_in_genre = movie_mask(df_co, col='genres', mask=genre_mask)
        df_co = df_co[df_co.movie.isin(films_in_genre)]

    if actor_mask:
        films_in_genre = movie_mask(df_co, col='actor', mask=actor_mask)
        df_co = df_co[df_co.movie.isin(films_in_genre)]

    return df_co


@st.cache
def get_graph(df_main,df_actor_all,actor_mask =None, year_threshold = None):
    df = get_co_occurrence_frame(df_main,df_actor_all,actor_mask = actor_mask,pop_mask=10)
    if year_threshold:
        df = df[df.release_year >= year_threshold]
    net = NetworkGeneration(df['actor'].dropna())
    net.fit_transform()
    return net


def get_no_edges_popularity(graph):
    df = pd.DataFrame(graph.no_edge_df.groupby('source')['target'].agg(lambda x: x.to_list()))
    df['no_edge_count'] = df['target'].apply(lambda x: len(x))
    return df


def get_edges_popularity(graph, cast):
    first_merge = pd.merge(graph.edge_df,cast[['actor','popularity_actor']],left_on='source',right_on='actor',how='left')
    second_merge = pd.merge(first_merge,cast[['actor','popularity_actor']],left_on='target',right_on='actor',how='left')
    second_merge = second_merge.drop(columns = ['actor_x','actor_y'])
    second_merge = second_merge.rename(columns={'popularity_actor_x':'source_popularity','popularity_actor_y':'target_popularity'})
    second_merge = second_merge.drop_duplicates(subset=['source','target'])
    second_merge['node_plot'] = second_merge['source']+' & '+second_merge['target']
    return second_merge


@st.cache
def get_embeddings(paths=[], drop_full_embeddings=True):
    dfs = [pd.read_csv(path, index_col=0) for path in paths]
    df = pd.concat(dfs)

    if drop_full_embeddings:
        return df[df.columns[:6]]
    else:
        return df


light = 'rgb(131,199,161)'
medium = 'rgb(6,180,226)'
dark = 'rgb(3,37,65)'

@st.cache
def get_data_frames():
    df = pd.read_json(os.getcwd()+'/data/main/tmdb_data_main.json')
    df.release_date = pd.to_datetime(df.release_date)
    cast = get_person_data_new(os.getcwd()+'/data/cast_crew/cast_3.jsonl')
    cast = cast[cast.adult == False]
    cast = cast[['tmdb_id', 'actor', 'popularity_actor', 'profile_path', 'character']]
    df_actor_all = pd.merge(cast, df, left_on='tmdb_id', right_on='tmdb_id', how='right').dropna()
    return df, cast, df_actor_all


df, cast, df_actor_all = get_data_frames()


def get_nounchunks(df, noun_chunk_col, noun):
    """
    Args:
        DataFrame with spaCy features, the column containing spaCy noun chunks and the noun to analyse
    Returns:
        DataFrame containing the noun chunks for that noun
    """

    df_nc = [i for list_ in df[noun_chunk_col].to_list() for i in list_]
    df_nc = pd.DataFrame(df_nc)
    df_nc.columns = ['noun', 'noun_chunk']
    df_nc['chunk_length'] = df_nc['noun_chunk'].map(lambda x: len(x))
    df_nc = df_nc[df_nc['noun'] == noun]
    df_nc = df_nc.sort_values(by='chunk_length', ascending=False)
    df_nc = df_nc[['noun', 'noun_chunk']]

    return df_nc


def add_noun(df,noun):
    word_l = list(df.word)
    for i in word_l:
        if i.split()[-1] != noun:
            word_l[word_l.index(i)] = word_l[word_l.index(i)]+' '+noun
    df.word = word_l

    return df


def getAllNounChunks(df, noun_col, noun_chunk_col, chunk_token, stop_nouns=[], top_nouns=20):
    """
    Args:
        DataFrame with spaCy features, the noun column, the noun chunk column, the number of grams needed and
        any nouns we want to exclude
    Returns
        DataFrame containing the counts of the most common tokens
    """

    mcn = list(most_common_tokens(df[noun_col], token=1).head(top_nouns)['word'])
    mcn = [i for i in mcn if i not in stop_nouns]  # exclude any stop nouns
    master_noun_chunks = []  # empty list for the noun chunks to be appended to

    # loop through each noun
    for noun in mcn:
        noun_chunks = get_nounchunks(df, noun_chunk_col, noun)['noun_chunk']  # get all noun chunks for noun
        try:
            # get most common noun chunks from all noun chunks
            mc_noun_chunks = most_common_tokens(noun_chunks, token=chunk_token)
            # adds the noun to the end of a noun chunk if its missing
            mc_noun_chunks = add_noun(mc_noun_chunks, noun)
            # append top n most common noun chunks to list
            master_noun_chunks.append(mc_noun_chunks)
        except:
            print(noun)

            pass

    return pd.concat([master_noun_chunks[i] for i in range(len(master_noun_chunks))])


def visualise_nounchunks(noun_chunk_df,colour_use):
    noun_chunk_df["all_nouns"] = ""  # empty string in order to have a single root node
    noun_chunk_df['noun'] = noun_chunk_df['word'].apply(lambda x: x.split(" ")[-1])

    fig = px.treemap(noun_chunk_df, path=['all_nouns', 'noun', 'word'],
                     values='count', hover_data=['count'], color='count', color_continuous_scale=colour_use)

    fig.update_layout(
        autosize=False,
        width=800,
        height=800)

    #     fig.update_traces(marker_colorscale = 'Blues')
    fig.update_traces(hovertemplate=None)
    # fig.show()
    return fig











def streamlit_init():

    st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
        unsafe_allow_html=True)
    query_params = st.experimental_get_query_params()
    tabs = ["Home", "Exploratory Analysis", 'Actor Filmography', "Actor Co-Occurrence", "Semantics", "Find Similar Descriptions"]

    im = Image.open('./assets/tmdb_logo_s.png')

    st.image(im.resize((int(im.size[0] / 2), int(im.size[1] / 2)), 0))

    if "tab" in query_params:
        active_tab = query_params["tab"][0]
    else:
        active_tab = "Home"

    if active_tab not in tabs:
        st.experimental_set_query_params(tab="Home")
        active_tab = "Home"

    li_items = "".join(
        f"""
        <li class="nav-item">
            <a class="nav-link{' active' if t == active_tab else ''}" href="/?tab={t}">{t}</a>
        </li>
        """
        for t in tabs
    )
    tabs_html = f"""
        <ul class="nav nav-tabs">
        {li_items}
        </ul>
    """

    st.markdown(tabs_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    return active_tab


active_tab = streamlit_init()

if active_tab == "Home":
    st.write('ed')

elif active_tab == "Exploratory Analysis":
    time_response = st.selectbox('Do you want analyse a period or a specific year?', ('A period', 'A specific decade', 'A specific year'))

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
        decades = st.selectbox('Select specific decade:', ('60s','70s','80s','90s','00s','10s','20s'))
        decade_dic = {'60s':(1960,1970),
                      '70s':(1970,1980),
                      '80s':(1980,1990),
                      '90s':(1990,2000),
                      '00s':(2000,2010),
                      '10s':(2010,2020),
                        '20s':(2020,2022)}
        df = df[(df.release_date > str(decade_dic[decades][0])) & (df.release_date < str(decade_dic[decades][1]))]
        year_string = 'in the '+decades

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
                                            barmode='stack', barnorm=barnorm,width=1600, height=650))

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
    actor_merge = pd.merge(actor_value_counts, actor_films,left_index=True,right_index=True,how='inner').reset_index()
    if actor_merge.shape[0] > 50:
        actor_merge = actor_merge.head(50)

    actor_merge.columns = ['actor','count','films']

    busiest_actor_fig = px.bar(actor_merge, x=actor_merge['actor'], y=actor_merge['count'])
    busiest_actor_fig.update_traces(marker_color=light)

    st.plotly_chart(plotly_streamlit_layout(plotly_streamlit_texts(busiest_actor_fig,x_title=None,y_title=None),width=1600, height=650))

    st.subheader(f'Busiest Actors {year_string} : Film Titles')
    c1,c2,c3 =st.beta_columns((0.5,2,0.4))
    actor_merge = actor_merge.explode('films')
    actor_merge = pd.merge(actor_merge,df[['budget','movie','popularity','vote_average']],how='left',left_on='films',right_on='movie')
    busiest_actor_fig_2 = px.sunburst(actor_merge, path=['actor', 'films'],values='budget',color='vote_average',color_continuous_scale=\
                                      px.colors.sequential.YlGnBu)

    busiest_actor_fig_2.update_layout(width=1000, height=1000, showlegend=False)
    # busiest_actor_fig_2.update_traces(marker=dict(colors=px.colors.qualitative.Pastel2))
    busiest_actor_fig_2.update_traces(textfont_size=12, marker=dict(line=dict(color='#000000', width=0.5)))
    busiest_actor_fig_2.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    c2.plotly_chart(busiest_actor_fig_2)

elif active_tab == "Actor Filmography":

    actor_list = df_actor_all.actor.unique()
    actor_list.sort()
    actor_data = st.selectbox('Select which actor to analyse', actor_list,index=0)
    df_actor_specific_frame = get_actor_frame(df_actor_all, actor_data)


    c1,mid,c2 = st.beta_columns((1,0.2,2))
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
    genres_in_pie.update_layout( margin=dict(l=20, r=20, t=20, b=20))

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
    texts = get_data_from_dic(df_actor_specific_frame,'keywords','name',string_return=True)

    image_mask = np.array(actor_image)
    wordcloud = WordCloud(font_path=streamlit_font_path, background_color="white", contour_color='black',
                          prefer_horizontal=True,width=1600,height=400,
                         mode="RGBA", max_words=2000).generate_from_text(texts)

    # image_colors = ImageColorGenerator(image_mask)
    plt.figure(figsize=(20, 10), dpi=2400)
    plt.imshow(wordcloud)
    plt.axis("off")
    fig1 = plt.gcf()

    st.pyplot(fig1)


















    st.header('Interactive Analysis')
    my_expander = st.beta_expander(label='Click for info')
    c1, c2 = st.beta_columns((2,2))
    x_ax = c1.selectbox('X Axis', ('Revenue', 'Budget', 'Release Date', 'Film Popularity', 'Review Average'), index=2)
    y_ax = c1.selectbox('Y Axis', ('Revenue', 'Budget', 'Release Date', 'Film Popularity', 'Review Average'), index=0)
    colour = c2.selectbox('Colour by', ('Revenue', 'Budget', 'Release Date', 'Film Popularity', 'Review Average'), index=4)
    size = c2.selectbox('Size by', ('Revenue', 'Budget', 'Release Date', 'Film Popularity', 'Review Average'), index=1)

    film_dic = {'Revenue': 'revenue',
                'Budget': 'budget',
                'Release Date': 'release_date',
                'Film Popularity': 'popularity',
                "Review Average": 'vote_average'}

    film_time = px.scatter(df_actor_specific_frame, x=film_dic[x_ax], opacity=0.7, size=film_dic[size]
                           , y=film_dic[y_ax],color=film_dic[colour],hover_name='movie',color_continuous_scale='ylgnbu')
    # film_time.update_marker()
    # film_time.update_traces(marker=dict(size=40))

    st.plotly_chart(plotly_streamlit_texts(plotly_streamlit_layout(film_time, height=650, width=1400),
                                           x_title=film_dic[x_ax],y_title=film_dic[y_ax]))

    st.header('Reviews')

    film_to_review = st.selectbox('Choose Film',df_actor_specific_frame.movie.unique())

    df_review = df_actor_specific_frame[df_actor_specific_frame.movie == film_to_review]

    reviews = get_data_from_dic(df_review, 'reviews', 'content', string_return=False)
    for i in range(len(reviews)):
        st.write(reviews[i])

elif active_tab == "Actor Co-Occurrence":
    # actor_list = df_actor_all.actor.unique()
    # actor_list.sort()
    # actor_data = st.selectbox('Select which actor to analyse', actor_list, index=2680)
    # df_actor_specific_frame = get_actor_frame(df_actor_all, actor_data)
    st.header('Actor Co-Occurence')
    net = get_graph(df, df_actor_all, year_threshold=None)

    df_edges = get_edges_popularity(net, cast)
    df_edges = df_edges.sort_values(by=['source'], ascending=False)

    fig = px.scatter(df_edges, x='target', y='source', size='edge_frequency', hover_name='node_plot',
                     color='edge_frequency', color_continuous_scale='YlGnBu')

    st.plotly_chart(plotly_streamlit_texts(plotly_streamlit_layout(fig, height=1600, width=1600), x_title=None,
                                           y_title=None))
    st.header('Films starred in together')
    c1,c2 = st.beta_columns((1,1))
    actors = list(df_actor_all.actor.unique())
    actors.sort()
    actor_1 = c1.selectbox('Actor 1:', actors )
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
        st.image(get_poster(poster_urls[0]), width=700)

    elif width_needed == 2:
        c1, c2 = st.beta_columns((5, 5))
        containers = [c1,c2]
        for i in range(len(poster_urls)):
            containers[i].image(get_poster(poster_urls[i]),width=550)

    elif width_needed == 3:
        c1, c2, c3 = st.beta_columns((4, 4, 4))
        containers = [c1, c2, c3]
        for i in range(len(poster_urls)):
            containers[i].image(get_poster(poster_urls[i]), width=450)

    elif width_needed == 4:
        c1, c2, c3, c4 = st.beta_columns((3, 3, 3, 3))
        containers = [c1, c2, c3, c4]
        for i in range(len(poster_urls)):
            containers[i].image(get_poster(poster_urls[i]), width=385)

    else:

        c1,c2,c3,c4,c5,c6,c7,c8 = st.beta_columns((2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1))
        containers = [c1,c2,c3,c4,c5,c6,c7,c8]
        for i in range(len(poster_urls)):
            containers[i].image(get_poster(poster_urls[i]),width=190)

elif active_tab == "Semantics":

    df_main = pd.read_json('./data/main/tmdb_data_main.json')
    df_main = df_main[['tmdb_id', 'movie', 'overview', 'genres', 'genres_new', 'poster_path',
                       'popularity', 'release_year', 'vote_average', 'budget', 'revenue']]

    st.header('Exploring Film Descriptions')

    film_list = df_main.movie.unique()
    film_list.sort()
    film_data = st.selectbox('Select film', film_list)

    df_mask = df_main[df_main.movie == film_data].reset_index()
    c1,c2 = st.beta_columns((3,4))
    c1.image(get_poster(df_mask.poster_path.unique()[0]))
    c2.subheader(df_mask.overview[0])
    c2.write(f'Released : {df_mask.release_year[0]}')
    c2.write(f'Genres : {str(df_mask.genres[0]).replace("[","").replace("]","")}')


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

    c1, c2 = st.beta_columns((2,2))
    dimensions = c1.selectbox('Components', ('2','3'),index=0)
    size = c2.selectbox('Size by', ('Popularity', 'Budget', 'Revenue', 'Vote Average','No Sizing'), index=4)

    size_dic = {'Vote Average': 'vote_average', 'Popularity': 'popularity',
                'Budget':'budget', 'Revenue': 'revenue', 'No Sizing': None}

    if dimensions == '2':

        fig = px.scatter(df_semantic_plot[df_semantic_plot.popularity >= 5], x='ts_2_x', y='ts_2_y',
                     color='genres_new', opacity=0.8,size=size_dic[size], hover_name='plot_data')

    else:
        fig = px.scatter_3d(df_semantic_plot[df_semantic_plot.popularity >= 5], x='ts_3_x', y='ts_3_y',z='ts_3_z',
                     color='genres_new', opacity=0.8,size=size_dic[size], hover_name='plot_data')

    st.plotly_chart(plotly_streamlit_layout(fig, height=1000, width=1000))

    st.header('Part of Speech Analysis')

    genre_analyse = st.selectbox('Choose genre', ('Action', 'Comedy', 'Drama', 'SciFi', 'Horror', 'Crime'), index=4).lower()
    st.subheader(f'Most common nouns used in {genre_analyse.capitalize()}')

    options_dic = {'action':'Oranges',
                   'comedy':'Blues',
                   'drama':'Greens',
                   'scifi':'Reds',
                   'horror':'Purples',
                   'crime':'Blues'}

    path = os.getcwd()+f'/data/nouns/{genre_analyse}.txt'

    with open(path) as f:
        x = pd.DataFrame(json.loads(f.read()))

    df_nouns = getAllNounChunks(x, 'spaCy_nouns', 'spaCy_noun_chunk', chunk_token=2, top_nouns=60)

    fig = visualise_nounchunks(df_nouns, colour_use=options_dic[genre_analyse])

    st.plotly_chart(plotly_streamlit_layout(fig,width=1500,height=800))


elif active_tab == 'Find Similar Descriptions':

    def get_vector_from_text(user_text, model_instance):
        vectorised_text = model_instance.fit_transform([user_text], reduce=False)
        vectorised_text_array = np.array(vectorised_text.T.iloc[1:][0])
        return vectorised_text_array

    @st.cache
    def get_full_embeddings():
        dfs = []
        parts = ['3000', '6000', '9000', '12000']
        for i in parts:
            df = pd.read_csv(f'./data/overview_embeddings/embed_full_{i}.csv')
            dfs.append(df)
        return pd.concat(dfs).set_index('tmdb_id').drop_duplicates()


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
    @st.cache
    def get_encoder():
        use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
        return use

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
        containers[i].image(get_poster(poster_paths[i]), width=450)
        containers[i].subheader(overviews[i])



else:
    st.error("Something has gone terribly wrong.")
