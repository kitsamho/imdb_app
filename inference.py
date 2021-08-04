import transformers
from simpletransformers.classification import MultiLabelClassificationModel
import torch
import numpy as np
import pandas as pd


model_path = '/Users/sam.ho/Downloads/checkpoint-1535-epoch-5'

inference_model = MultiLabelClassificationModel('distilbert',model_path, use_cuda=False)

if __name__ == '__main__':
    preds, outputs = inference_model.predict(['Dave Smith was a happy go lucky kind of guy until he met Sally one day. \
                                              From that point they fell in love and moved in together and bought a dog'])

    labels = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama',
              'Family', 'Horror', 'Romance', 'Science Fiction', 'Thriller']

    sub_df = pd.DataFrame(outputs, columns=labels).T
    sub_df = sub_df.sort_values(by=0, ascending=False)
    print(sub_df)
    print(inference_model)



