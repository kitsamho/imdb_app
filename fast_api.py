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

model_path = '/Users/saho/Documents/sam/checkpoint-1535-epoch-5'
inference_model = MultiLabelClassificationModel('distilbert',model_path, use_cuda=False)
# text_input = ['Hoping to spend Christmas with his estranged wife, detective John McClane arrives in LA. However, he \
# learns about a hostage situation in an office building and his wife is one of the hostages']
# predictions, _ = inference_model.predict([text_input])
# print(predictions[0])

labels_dic = {'Action'	'Adventure'	'Animation'	'Comedy'	'Crime'	'Drama'	'Family'	'Fantasy'	'Horror'	'Romance'
'Science' 'Fiction'	'Thriller'}

@app.get("/")
def get_predictions(review):
    # model_path = '/Users/saho/Documents/sam/checkpoint-1535-epoch-5'
    # inference_model = MultiLabelClassificationModel('distilbert', model_path, use_cuda=False)
    preds, outputs = inference_model.predict([review])

    preds, outputs = model.predict(test)

    sub_df = pd.DataFrame(outputs, columns=genre_include)


    return {"Result": predictions}


