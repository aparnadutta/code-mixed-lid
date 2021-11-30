# Code is based on https://github.com/AU-DIS/LSTM_langid
from typing import Optional, Callable

import torch
from torchtext.data import sentencepiece_numericalizer, load_sp_model

from run_language_identifier import run_training
from lstm_model import LSTMLIDModel


def load_LSTM_model(pretrained_model_path: Optional[str], subword_to_idx: Callable, lang_to_idx: dict,
                    hidden_dim, embedding_dim, num_lstm_layers):
    if pretrained_model_path is not None:
        model_dict = torch.load(pretrained_model_path)
        LSTM_model = LSTMLIDModel(model_dict['subword_to_idx'], model_dict['lang_to_idx'],
                                  model_dict['embedding_dim'], model_dict['hidden_dim'], model_dict['layers'])
        LSTM_model.load_state_dict(model_dict['model_state_dict'])

    else:
        LSTM_model = LSTMLIDModel(subword_to_idx=subword_to_idx, lang_to_idx=lang_to_idx,
                                  hidden_dim=hidden_dim, embedding_dim=embedding_dim, layers=num_lstm_layers)
    return LSTM_model


def main(pretrained_model, epochs, weight_decay, batch_size, lr, optimizer):

    training_params = optimizer, weight_decay, lr, batch_size, epochs
    numericalizer = sentencepiece_numericalizer(load_sp_model('./spm_user.model'))

    subword_to_idx = numericalizer
    lang_to_idx = {'bn': 0, 'univ': 1, 'en+bn_suffix': 2, 'undef': 3, 'hi': 4, 'ne': 5, 'en': 6, 'acro': 7,
                   'ne+bn_suffix': 8}

    lstm_model = load_LSTM_model(pretrained_model_path=pretrained_model,
                                 subword_to_idx=subword_to_idx,
                                 lang_to_idx=lang_to_idx,
                                 hidden_dim=HIDDEN_DIM,
                                 embedding_dim=EMBEDDING_DIM,
                                 num_lstm_layers=NUM_LSTM_LAYERS)
    to_train = pretrained_model is None

    run_training(lstm_model, training_params, to_train)


PRETRAINED_MODEL = None
EPOCHS = 1
SEED = 42
HIDDEN_DIM = 100
EMBEDDING_DIM = 75
NUM_LSTM_LAYERS = 2
OPTIMIZER = 'SGD'
LR = 0.1
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 64

if __name__ == "__main__":
    main(pretrained_model=PRETRAINED_MODEL,
         epochs=EPOCHS,
         weight_decay=WEIGHT_DECAY,
         batch_size=BATCH_SIZE,
         lr=LR,
         optimizer=OPTIMIZER)
    print(f'PRETRAINED_MODEL = {PRETRAINED_MODEL}\n'
          f'EPOCHS = {EPOCHS}\n'
          f'HIDDEN_DIM = {HIDDEN_DIM}\n'
          f'EMBEDDING_DIM = {EMBEDDING_DIM}\n'
          f'NUM_LSTM_LAYERS = {NUM_LSTM_LAYERS}\n'
          f'OPTIMIZER = {OPTIMIZER}\n'
          f'LR = {LR}\n'
          f'WEIGHT_DECAY = {WEIGHT_DECAY}\n'
          f'BATCH_SIZE = {BATCH_SIZE}')
