import pandas as pd
import re

def clean_tweet(tweet):
    result = re.sub(r"http\S+", "", tweet)
    tweet_words = [x for x in result.split() if
                   not x.startswith('#') and not x.startswith('@')]
    return ' '.join(tweet_words)


def get_data(inp_file):
    df = pd.read_csv(inp_file, encoding="ISO-8859-1")
    df['cleaned_tweet'] = df['tweet'].map(lambda x: clean_tweet(x))
    df['label'] = df['label'].map(lambda x: make_label(x))
    return df


def make_label(label):
    if label == 4:
        label = 1
    return label
