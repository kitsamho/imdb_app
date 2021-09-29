
from simpletransformers.classification import MultiLabelClassificationModel

import pandas as pd

from fastapi import FastAPI
from multiprocessing import set_start_method


movie_classification = FastAPI(
    title="Movie Genre Classification",
    description="FastAPI that used a fine-tuned distilBERT multi-label model to classify film descriptions by genre",
    version="0.1")

lethal_weapon = "The crew of a spacecraft, Nostromo, intercept a distress signal from a planet and set out to investigate it. However, to their horror, they are attacked by an alien which later invades their ship"

model_path = '/Users/saho/Documents/sam/checkpoint-744-epoch-3'

labels = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Science Fiction', 'Thriller']


def load_model(model_path):
    model = MultiLabelClassificationModel('distilbert', model_path, use_cuda=False)
    return model


model = load_model(model_path)


def inference(texts):

    pred_label, probabilities = model.predict([texts])
    df = pd.DataFrame(dict(zip(labels, probabilities[0])), index=[0])
    df = df.T.reset_index()
    df = df.sort_values(by=0, ascending=False)

    results_api = dict(zip(df['index'], df[0]))
    # print(results_api)
    return results_api

try:
    set_start_method('spawn')
except RuntimeError:
    pass

@movie_classification.get("/classify_description")
def classification(text:str):
    """function to classify article using a deep learning model.
    Returns:
        [type]: [description]
    """
    results_api = inference(text)
    return results_api


















