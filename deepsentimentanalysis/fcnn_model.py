import gc
import sys

import numpy as np
import pandas as pd
from gensim import models
from gensim.models.doc2vec import TaggedDocument
from keras import Sequential
from keras.layers import Dense, Dropout
from nltk import TweetTokenizer

from .model import Model


class FCNNModel(Model):
    """
    Class for Fully Connected model which works on Doc vectors

    This class implements functionality for implementing the functionality of performing
    sentimental analysis on tweets on using a fully connected neural network on sentence
    vectors obtained from tweets
    """

    def __init__(self, data_dimensions, save_path=None, batch_size=32, num_epochs=2,
                 prerequisite_save_path=None):
        super(FCNNModel, self).__init__(data_dimensions, save_path, batch_size,
                                        num_epochs, prerequisite_save_path)
        self.__generate_model()
        self._d2v_model = None

    def _augment_data(self, data_df):
        print 'Augmenting data'
        if not self._d2v_model:
            if self.prerequisite_save_path:
                d2v_model = models.Doc2Vec.load(self.prerequisite_save_path)
            else:
                sys.stderr.write('Either generate the doc2vec model or '
                                 'give path for the saved model\n')
                sys.exit(-1)
        data_df['docvec'] = data_df['id'].map(lambda x: d2v_model.docvecs[str(x)])
        print 'Done'
        print data_df.head()
        return np.asarray(data_df.docvec.tolist()), np.asarray(data_df.label)

    def generate_prerequisites(self, train_df, test_df):
        tokenizer = TweetTokenizer()

        def tokenize(tweet):
            """
            Tokenise a given tweet and return the tokens
            :param tweet: tweet to be tokenized
            :return: tokens from the tweet if successful empty list otherwise
            """
            return tuple(tokenizer.tokenize(tweet))

        data = pd.concat([train_df, test_df], axis=0)
        sentences = []
        for index, row in data.iterrows():
            tokens = tokenize(row['tweet'])
            if tokens:
                sentences.append(TaggedDocument(tokens, tags=[str(row['id'])]))
        print(len(sentences))
        del data
        gc.collect()
        print 'data prepared'
        self._d2v_model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1,
                                         vector_size=self.data_dim[0])
        self._d2v_model.build_vocab(sentences)
        print 'Docvecs training started'
        self._d2v_model.train(sentences, total_examples=self._d2v_model.corpus_count,
                              epochs=self._d2v_model.iter)
        print 'Docvecs training Finished'
        del sentences
        gc.collect()
        if self.prerequisite_save_path:
            self._d2v_model.save(self.prerequisite_save_path)
        gc.collect()

    def __generate_model(self):
        self.model = Sequential()
        self.model.add(Dense(250, activation='tanh', input_shape=self.data_dim))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(64, activation='tanh'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
