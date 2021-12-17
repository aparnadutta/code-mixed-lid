import torch
import torch.nn as nn
from src.lid_model import LIDModel


DROPOUT = 0.4


class LSTMLIDModel(LIDModel):
    def __init__(self, subword_to_idx, lang_to_idx, embedding_dim=300, hidden_dim=300, layers=1):
        super(LSTMLIDModel, self).__init__(subword_to_idx, lang_to_idx)
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, self.lang_set_size)
        self.dropout = nn.Dropout(DROPOUT)
        self.to(self.device)

    def forward(self, sentence):
        embed = self.embedding(sentence)
        embed = self.dropout(embed)
        outputs, _ = self.lstm(embed)
        outputs = self.linear(outputs)
        return outputs.transpose(1, 2)

    def save_model(self, fileending=""):
        """Saves a dict containing statedict and other required model parameters and adds it as artifact
        Arguments:
        """
        required_model_information = {'model_state_dict': self.state_dict(),
                                      'embedding_dim': self.embedding_dim,
                                      'hidden_dim': self.hidden_dim,
                                      'layers': self.num_layers}
        fname = "scratch_models/trained_LID_model" + fileending + ".pth"
        torch.save(required_model_information, fname)
