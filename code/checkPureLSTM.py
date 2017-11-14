import pickle
from sklearn.metrics import precision_recall_fscore_support

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from config import config
from utils import get_data

test_inp = config['test_input_file']
token_pickle = config['token_pickle_pure']
model_loc = config['lstm_model_loc']
with open(token_pickle, 'rb') as handle:
    tokenizer = pickle.load(handle)
print('tokenizer loaded')

test_df = get_data(test_inp)
test_df.drop(test_df[test_df.cleaned_tweet.apply(
    lambda x: len(tokenizer.texts_to_sequences([x])[0])) == 0].index,
             inplace=True)
x_test = test_df.cleaned_tweet.values
y_test = test_df.label.values
print('data loaded')
model = load_model(model_loc)
max_len = config['pad_len']
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, max_len)

predictions = model.predict(x_test)
pred = [round(x[0]) for x in predictions]
prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred,
                                                   average='binary')
print('recall: ', round(rec * 100, 3))
print('precision: ', round(prec * 100, 3))
print('f1score ', round(f1 * 100, 3))
