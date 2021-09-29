# visuals
def plotly_streamlit_layout(fig, barmode=None, barnorm=None, height=None, width=None):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      barmode=barmode,
                      barnorm=barnorm,
                      height=height,
                      width=width)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig.update_layout(margin=dict(l=50, r=50, b=50, t=50, pad=2))
    fig.update_layout(bargap=0.03)

    return fig


# plotly text formatting wrapper
def plotly_streamlit_texts(fig, x_title, y_title):
    fig.update_layout(yaxis=dict(title=y_title, titlefont_size=8, tickfont_size=8),
                      xaxis=dict(title=x_title, titlefont_size=8, tickfont_size=8))
    return fig
