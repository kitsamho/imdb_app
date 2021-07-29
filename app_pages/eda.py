import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow_hub as hub
from app_dependencies.visuals import plotly_streamlit_layout, plotly_streamlit_texts

# plotly colour palettes
light = 'rgb(131,199,161)'
medium = 'rgb(6,180,226)'
dark = 'rgb(3,37,65)'

def write(df, df_actor_all):

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
    return

