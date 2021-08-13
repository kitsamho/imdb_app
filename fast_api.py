import transformers
from simpletransformers.classification import MultiLabelClassificationModel
import torch
import numpy as np
import pandas as pd
import streamlit as st
import logging
from fastapi import FastAPI

app = FastAPI(
    title="Movie Genre Classification",
    description="FastAPI that used a fine-tuned distilBERT multi-label model to classify film descriptions by genre",
    version="0.1",
)

lethal_weapon = "Ellen Ripley is sent back to the planet LV-426 to establish contact with a terraforming colony. Once there, she encounters the Alien Queen and her offspring and has to fight them to survive."

model_path = '/Users/saho/Documents/sam/checkpoint-1535-epoch-5'

model = MultiLabelClassificationModel('distilbert',model_path, use_cuda=False)

labels = ['Action', 'Adventure', 'Animation', 'Comedy',
                'Crime', 'Drama', 'Family', 'Horror',
                'Romance','Science Fiction','Thriller']

Action	Adventure	Comedy	Crime	Drama	Family	Horror	Romance	Science Fiction	Thriller
# @app.get("/inference")
def inference(texts):

    pred_label, probabilities = model.predict([texts])

    df = pd.DataFrame(dict(zip(labels, probabilities[0])), index=[0])
    df = df.T.reset_index()
    df = df.sort_values(by=0, ascending=False)

    results_api = dict(zip(df['index'], df[0]))

    return results_api, pred_label, probabilities

if __name__ == '__main__':
    results_api, pred_label, probabilities = inference(lethal_weapon)
    print('here')






