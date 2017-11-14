import gc
from gensim import models
from keras.layers import Dense, Dropout
from keras.models import Sequential
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


train_inp = config['train_input_file']
test_inp = config['test_input_file']
model_loc = config['fcnn_model_loc']
train_df = get_data(train_inp)
print('data read')
data = []
labels = []
for index, row in train_df.iterrows():
    docvec = get_docvec(row['id'])
    if docvec is not None:
        data.append(docvec)
        labels.append(row['label'])
del train_df
gc.collect()
x_train = np.array(data)
y_train = np.array(labels)
del data
del labels
gc.collect()
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
model = Sequential()
model.add(Dense(250, activation='tanh',input_shape=x_train[0].shape))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
batch_size = 32
print('model training started')
model.fit(x_train, y_train, epochs=100, batch_size=batch_size,
          validation_data=(x_test, y_test))
model.save(model_loc)
print('model fitted')
