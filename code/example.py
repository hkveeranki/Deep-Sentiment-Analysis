from config import config
from fcnn_model import FCNNModel
from lstm_model import LSTMModel
from utils import get_data

train_inp = config['train_input_file']
test_inp = config['test_input_file']

train_df = get_data(train_inp)
test_df = get_data(test_inp)

print 'Data Read'
name = 'FCNN'
if name == 'FCNN':
    model = FCNNModel((config['d2v_size'],), save_path=config['fcnn_model_loc'],
                      prerequisite_save_path=config['doc2vec_model'], num_epochs=100)
else:
    model = LSTMModel(config['pad_len'], save_path=config['lstm_model_loc'],
                      prerequisite_save_path=config['tokenizer_pickle'])
model.generate_prerequisites(train_df, test_df)
model.train(train_df)
print model.test(test_df, saved_weights=model.save_path)
