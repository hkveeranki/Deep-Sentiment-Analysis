import gc
from gensim import models
from keras.models import load_model
import numpy as np
from config import config
from utils import get_data

d2v_model = models.Doc2Vec.load(config['doc2vec_model'])


def get_docvec(id):
    """
    Get doc2vec vector for given id
    :param x: id of the tweet
    :return: doc2vec vector of the tweet if successful None otherwise
    """
    try:
        return d2v_model.docvecs[str(id)]
    except:
        return None


test_inp = config['test_input_file']
model_loc = config['fcnn_model_loc']
test_df = get_data(test_inp)
print('data read')
data = []
labels = []
for index, row in test_df.iterrows():
    docvec = get_docvec(row['id'])
    if docvec is not None:
        data.append(docvec)
        labels.append(row['label'])
del test_df
gc.collect()
x_test = np.array(data)
y_test = np.array(labels)
del data
del labels
gc.collect()
print('data prepared')
gc.collect()
batch_size = 32
model = load_model(model_loc)
scores = model.evaluate(x_test, y_test)
print('acc: ', round(scores[1] * 100, 3))
