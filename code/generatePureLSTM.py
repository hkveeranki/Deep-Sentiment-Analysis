#!/bin/python
import gc
import pickle
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, \
    Flatten, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from utils import get_data
from config import config

data = []

token_pickle = config['token_pickle_pure']
model_loc = config['lstm_model_loc']
train_inp = config['train_input_file']
test_inp = config['test_input_file']
with open(token_pickle, 'rb') as handle:
    tokenizer = pickle.load(handle)
print('Tokenizer loaded')
train_df = get_data(train_inp)
# Dropping bad data
train_df.drop(train_df[train_df.cleaned_tweet.apply(
    lambda x: len(tokenizer.texts_to_sequences([x])[0])) == 0].index,
              inplace=True)
x_train = train_df.cleaned_tweet.values
y_train = train_df.label.values
del train_df
gc.collect()
test_df = get_data(test_inp)
# Dropping bad data
test_df.drop(test_df[test_df.cleaned_tweet.apply(
    lambda x: len(tokenizer.texts_to_sequences([x])[0])) == 0].index,
             inplace=True)
x_test = test_df.cleaned_tweet.values
y_test = test_df.label.values
del test_df
gc.collect()
print('data read')

max_len = config['pad_len']
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, max_len)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, max_len)
embed_dim = 128
lstm_out = 64
model = Sequential()
model.add(
    Embedding(config['num_words'], embed_dim, input_length=max_len))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out))
model.add(Dropout(0.5))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
batch_size = 32
print('model training started')
model.fit(x_train, y_train, epochs=2, batch_size=batch_size,
          validation_data=(x_test, y_test))
model.save(model_loc)
print('model fitted')
