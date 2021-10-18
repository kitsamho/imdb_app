import pandas as pd
import plotly.express as px
import streamlit as st
from app_dependencies.visuals import plotly_streamlit_layout, plotly_streamlit_texts

# plotly colour palettes
light = 'rgb(131,199,161)'
medium = 'rgb(6,180,226)'
dark = 'rgb(3,37,65)'

def write(df, df_actor_all):

    df = df[df.release_date >= '1980-01-01']

    st.title('Overview')
    c1, mid, c2 = st.columns((3,0.5,3))

    time_response = c1.selectbox('Do you want analyse a defined period or a specific decade / year?',
                                 ('A specific decade', 'A specific year'))

    if time_response == 'A specific year':
        years = list(set([i.year for i in df.release_date]))[1:]
        years.sort(reverse=True)
        year = c2.selectbox('Select specific year:', years)
        df = df[(df.release_date > str(year)) & (df.release_date < str(year + 1))]
        year_string = year

    else:
        decades = c2.selectbox('Select specific decade:', ('1980s', '1990s', '2000s', '2010s', '2020s'))
        decade_dic = {'1980s': (1980, 1990),
                      '1990s': (1990, 2000),
                      '2000s': (2000, 2010),
                      '2010s': (2010, 2020),
                      '2020s': (2020, 2022)}
        df = df[(df.release_date > str(decade_dic[decades][0])) & (df.release_date < str(decade_dic[decades][1]))]
        year_string = decades

    st.header(f'Budgets and Revenue: {year_string}')
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
                                            barmode='stack', barnorm=barnorm, width=1200, height=650))


    st.header(f'Genres and Films: {year_string}')
    genre_expander = st.expander(label='Click for info')
    genre_expander.write('This is a treemap of films and their respective genres. For each film you can investigate\
      budget/revenue/profit (size of leaf) and the average vote the film received on IMDB (hue of leaf).')

    genre_period = df.explode('genres').dropna(subset=['genres'])
    # genre_period = genre_period[genre_period.vote_average > 0]
    genre_period['profit'] = genre_period['revenue'] - genre_period['budget']

    genre_plot_size_by = st.selectbox('Size leaves by:', ('budget', 'revenue', 'profit'))

    genre_period = genre_period[(genre_period['vote_average'] != 0) & (genre_period['budget'] != 0) &
                                (genre_period['revenue'] != 0) & (genre_period['profit'] != 0)]

    plot_value = genre_plot_size_by.lower()
    genre_period_fig = px.treemap(genre_period, path=[px.Constant("Films"), 'genres', 'movie'],
                                  values=plot_value,
                                  color='vote_average',
                                  color_continuous_scale='YlGnBu')
    genre_period_fig.update_layout(width=1200, height=650,
                                   margin=dict(t=50, l=25, r=25, b=25))

    st.plotly_chart(genre_period_fig)

    st.header(f'Actors : {year_string}')
    film_count_expander = st.expander(label='Click for info')
    film_count_expander.write('This is a sunburst map of actors and the films they starred in. For each film you can investigate\
      budget/revenue/profit (size of segment) and the average vote the film received on IMDB (hue of segment).')


    actor_value_counts = df_actor_all.actor.value_counts()

    actor_films = df_actor_all.groupby('actor')['movie'].agg(lambda x: x.to_list())
    actor_merge = pd.merge(actor_value_counts, actor_films, left_index=True, right_index=True,
                           how='inner').reset_index()

    if actor_merge.shape[0] > 50:
        actor_merge = actor_merge.head(200)

    actor_merge.columns = ['actor', 'count', 'films']
    c1, c2, c3 = st.columns((0.5, 2, 0.4))
    actor_merge = actor_merge.explode('films')
    actor_merge = pd.merge(actor_merge, df[['revenue', 'movie','budget', 'popularity', 'vote_average']], how='left',
                           left_on='films', right_on='movie')
    
    actor_merge['profit'] = actor_merge['revenue'] - actor_merge['budget']

    actor_merge_plot_size_by = st.selectbox('Size segments by:', ('Budget', 'Revenue', 'Profit'))

    plot_value = actor_merge_plot_size_by.lower()

    busiest_actor_fig_2 = px.sunburst(actor_merge, path=['actor', 'films'], values=plot_value, color='vote_average',
                                      color_continuous_scale= \
                                          px.colors.sequential.YlGnBu)

    busiest_actor_fig_2.update_layout(width=800, height=800, showlegend=False)
    busiest_actor_fig_2.update_traces(textfont_size=12, marker=dict(line=dict(color='#000000', width=0.5)))
    busiest_actor_fig_2.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    st.plotly_chart(busiest_actor_fig_2)





