import pickle

import gc
import pandas as pd
from keras.preprocessing.text import Tokenizer

from utils import get_data
from config import config

train_inp = config['train_input_file']
test_inp = config['test_input_file']

train_df = get_data(train_inp)
test_df = get_data(test_inp)
data = pd.concat([train_df, test_df], axis=0)
del train_df
del test_df
gc.collect()
print('data prepared')
max_fatures = 20000
print('tokenizer started')
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['cleaned_tweet'])
with open('../models/tokenizer_pureLSTM.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('tokenizer fitted')
