
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import streamlit as st


model_path = '/Users/saho/Documents/sam/checkpoint-744-epoch-3'
labels_for_inference = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Science Fiction', 'Thriller']

@st.cache
def load_model(model_path):
    model = MultiLabelClassificationModel('distilbert', model_path, use_cuda=False)
    return model


model = load_model(model_path)


def inference(texts):

    pred_label, probabilities = model.predict([texts])
    df = pd.DataFrame(dict(zip(labels_for_inference, probabilities[0])), index=[0])
    df = df.T.reset_index()
    df = df.sort_values(by=0, ascending=False)
    results_api = dict(zip(df['index'], df[0]))
    return results_api,df

st.title('Film Genre Classification')


user_input = st.text_area('Write a film description')

# pred_label, probabilities = model.predict([user_input])
# st.write(user_input)

results_api, df = inference(user_input)
st.write(df)



# if __name__ == '__main__':
#     text = ['A woman has 24 hours to establish her innocence but has to face a mysterious establishment']
#     results_api, df = inference(text)
#     print(results_api)
#     print(df)



















