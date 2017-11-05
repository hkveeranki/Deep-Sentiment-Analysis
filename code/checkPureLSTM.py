import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from config import config
from utils import get_data

test_inp = config['test_input_file']
token_pickle = config['token_pickle_pure']
model_loc = config['model_loc_pure']

test_df = get_data(test_inp)
x_test = test_df.cleaned_tweet.values
y_test = test_df.label.values
print('data loaded')
with open(token_pickle, 'rb') as handle:
    tokenizer = pickle.load(handle)
print('tokenizer loaded')
model = load_model(model_loc)
max_len = config['pad_len']
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, max_len)

scores = model.evaluate(x_test, y_test)
print('acc: ', round(scores[1] * 100,3))
