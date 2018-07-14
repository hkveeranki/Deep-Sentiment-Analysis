"""
Module deep_sentiment_analysis
"""
__author__ = 'Hemanth Kumar Veeranki'
__version__ = '0.1.0'
config = {
    'train_input_file': '../data/train.csv',
    'test_input_file': '../data/test.csv',
    'tokenizer_pickle': '../models/tokenizer_pureLSTM.pickle',
    'lstm_model_loc': '../models/pureLSTM_model.h5',
    'doc2vec_model': '../models/tweet_model.doc2vec',
    'fcnn_model_loc': '../models/fcnn_model.h5',
    'pad_len': 60,
    'num_words': 50000,
    'd2v_size': 200
}
