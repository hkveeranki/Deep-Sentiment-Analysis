import gc
import pandas as pd
from gensim import models
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from config import config
from utils import get_data

tokenizer = TweetTokenizer()
stop_words = set(stopwords.words('english'))


def tokenize(tweet):
    """
    Tokenise a given tweet and return the tokens
    :param tweet: tweet to be tokenized
    :return: tokens from the tweet if successful empty list' otherwise
    """
    try:
        tweet = unicode(tweet.decode('utf-8').lower())
        tweet_tokens = tokenizer.tokenize(tweet)
        tweet_tokens = filter(lambda t: t not in stop_words, tweet_tokens)
        return tweet_tokens
    except:
        return []


train_inp = config['train_input_file']
test_inp = config['test_input_file']

train_df = get_data(train_inp)
test_df = get_data(test_inp)
data = pd.concat([train_df, test_df], axis=0)
del train_df
del test_df
gc.collect()
print('data read')
sentences = []
for index, row in data.iterrows():
    tokens = tokenize(row['cleaned_tweet'])
    if len(tokens) != 0:
        sentences.append(TaggedDocument(tokens, tags=[str(row['id'])]))
del data
gc.collect()
print('data prepared')
model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
model.build_vocab(sentences)
print('Training started')
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
print('Training Finished')
model.save('../models/tweet_model.doc2vec')
