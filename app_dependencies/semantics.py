import pandas as pd
import streamlit as st
import plotly.express as px

# semantics page
@st.cache
def get_embeddings(paths=[], drop_full_embeddings=True):
    dfs = [pd.read_csv(path, index_col=0) for path in paths]
    df = pd.concat(dfs)
    if drop_full_embeddings:
        return df[df.columns[:6]]
    else:
        return df


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


def add_noun(df, noun):
    word_l = list(df.word)
    for i in word_l:
        if i.split()[-1] != noun:
            word_l[word_l.index(i)] = word_l[word_l.index(i)] + ' ' + noun
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


def visualise_nounchunks(noun_chunk_df, colour_use):
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