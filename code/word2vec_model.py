from gensim.models.doc2vec import LabeledSentence
from gensim.models import Word2Vec, Doc2Vec
from nltk import TweetTokenizer
# from nltk.corpus import stopwords
from sklearn.utils import shuffle
import pandas as pd
from tqdm import tqdm

from config import config

tokenizer = TweetTokenizer()


# stop_words = set(stopwords.words('english'))


def tokenize(tweet):
    """
    Tokenize the given tweet and return it without mentions hash tags and urls
    :param tweet: tweet to be tokenized
    :return: tokens
    """
    # tweet = unicode(tweet.decode('utf-8').lower())
    tokens = tokenizer.tokenize(tweet)
    tokens = filter(lambda t: not t.startswith('@'), tokens)
    tokens = filter(lambda t: not t.startswith('#'), tokens)
    tokens = filter(lambda t: not t.startswith('http'), tokens)
    # tokens = filter(lambda t: not t in stop_words, tokens)
    return tokens


sentences = []

train_inp = config['train_input_file']
test_inp = config['test_input_file']


def get_df(inp_file):
    """
    Get the processed dataframe from the given file
    :param inp_file: name of the input file
    :return: processed dataframe
    """
    df = pd.read_csv(inp_file, sep='\t')
    df = shuffle(df)
    df['tokens'] = df['tweet'].map(tokenize)
    return df


if __name__ == "__main__":
    train_df = get_df(train_inp)
    test_df = get_df(test_inp)

    for index, row in tqdm(train_df.iterrows()):
        sentences.append(LabeledSentence(row['tokens'], [str(row['id'])]))
    for index, row in tqdm(test_df.iterrows()):
        sentences.append(LabeledSentence(row['tokens'], [str(row['id'])]))
    model = Doc2Vec(min_count=10)
    model.build_vocab(sentences)
    model.train(sentences, epochs=100, total_examples=len(sentences))
    model.save('../models/tweet_model.d2v')
    words = [x.words for x in sentences]
    model = Word2Vec(min_count=10)
    model.build_vocab(words)
    model.train(words, total_words=len(words),
                epochs=1000)
    model.save("../models/tweet_model.word2vec")
    print model.most_similar(':)')
