#!/bin/python
import gc
import pickle
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from utils import get_data
from config import config

data = []

token_pickle = config['token_pickle_pure']
model_loc = config['model_loc_pure']
train_inp = config['train_input_file']

train_df = get_data(train_inp)
x_train = train_df.cleaned_tweet.values
y_train = train_df.label.values
del train_df
gc.collect()
print('data read')

with open(token_pickle, 'rb') as handle:
    tokenizer = pickle.load(handle)
print('Tokenizer loaded')
max_len = config['pad_len']
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, max_len)
embed_dim = 128
lstm_out = 64
model = Sequential()
model.add(
    Embedding(len(tokenizer.word_index) + 1, embed_dim, input_length=max_len,
              dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
batch_size = 32
print('model training started')
model.fit(x_train, y_train, nb_epoch=1, batch_size=batch_size)
model.save(model_loc)
print('model fitted')
