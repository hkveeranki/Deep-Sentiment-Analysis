import re

import pandas as pd
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def clean_tweet(tweet):
    """Clean the given tweet and return it

    This function takes as input a tweet, removes urls, hash tags and stop words from the
    tweet and returns the remaining portion

    :param tweet: raw tweet data

    :return: remaining potion of the tweet

    """
    # Replace more than three occurrences
    # tweet = re.sub(r'([a-z])\1{2,}', r'\1\1', tweet)
    result = re.sub(r'http\S+', '', tweet)
    tweet_words = [x for x in result.split() if
                   not x.startswith('#') and not x.startswith('@') and
                   x not in stop_words]
    return ' '.join(tweet_words)


def get_data(inp_file):
    """ returns the data from the given file as pandas data frame

    This function loads the data from file into pandas data frame and process the data

    :param inp_file: path of the input file

    :return: pandas data frame containing processed data

    """
    print 'Reading', inp_file
    df = pd.read_csv(inp_file, encoding="ISO-8859-1")
    df['tweet'] = df['tweet'].map(lambda x: clean_tweet(x))
    df['label'] = df['label'].map(lambda x: make_label(x))
    print 'Done'
    return df


def make_label(label):
    """ map the labels to 0 or 1

    Given a label from the input data it converts into labels suitable for training the
    model

    :param label: raw label 0 or 4

    :return: return 1 if label is 4 0 otherwise

    """
    if label == 4:
        label = 1
    return label
