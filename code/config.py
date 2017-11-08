config = {}
config['train_input_file'] = '../input/train.csv'
config['test_input_file'] = '../input/test.csv'
config['pad_len'] = 150
config['token_pickle_pure'] = '../models/tokenizer_pureLSTM.pickle'
config['lstm_model_loc'] = '../models/pureLSTM_model.h5'
config['doc2vec_model'] = '../models/tweet_model.doc2vec'
config['fcnn_model_loc'] = '../models/fcnn_model.h5'