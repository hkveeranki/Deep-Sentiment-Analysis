import pandas as pd
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def clean_tweet(tweet):
    """
    Clean the given tweet to remove urls, mentions hashtags and stopwords
    :param tweet: raw tweet data
    :return: cleaned tweet
    """
    # Replace more than three occurrences
    # tweet = re.sub(r'([a-z])\1{2,}', r'\1\1', tweet)
    result = re.sub(r'http\S+', '', tweet)
    tweet_words = [x for x in result.split() if
                   not x.startswith('#') and not x.startswith('@')]
    return ' '.join(tweet_words)


def get_data(inp_file):
    """
    Load the data from file into pandas dataframe and process the data
    :param inp_file: path of the input file
    :return: pandas dataframe containing processed data
    """
    df = pd.read_csv(inp_file, encoding="ISO-8859-1")
    df['cleaned_tweet'] = df['tweet'].map(lambda x: clean_tweet(x))
    df['label'] = df['label'].map(lambda x: make_label(x))
    return df


def make_label(label):
    """
    map the labels to 0 or 1
    :param label: raw label 0 or 4
    :return: return 1 if label is 4 0 otherwise
    """
    if label == 4:
        label = 1
    return label
