import torch
import torch.nn as nn
from lid_model import LIDModel
import tempfile
import os

DROPOUT = 0.4


class LSTMLIDModel(LIDModel):
    def __init__(self, subword_to_idx, lang_to_idx, embedding_dim=300, hidden_dim=300, layers=2):
        super(LSTMLIDModel, self).__init__(subword_to_idx, lang_to_idx)
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, self.lang_set_size)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(DROPOUT)
        self.to(self.device)

    def init_hidden(self):
        return torch.randn(2*self.num_layers, 1, self.hidden_dim), torch.randn(2*self.num_layers, 1, self.hidden_dim)

    def forward(self, sentence):
        embed = self.embedding(sentence)
        embed = self.dropout(embed)
        outputs, _ = self.lstm(embed, self.hidden)
        outputs = self.linear(outputs)
        return outputs.squeeze()

    def save_model(self, fileending=""):
        """Saves a dict containing statedict and other required model parameters and adds it as artifact
        Arguments:
        """
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        required_model_information = {'subword_to_idx': self.subword_to_idx,
                                      'lang_to_idx': self.lang_to_idx,
                                      'model_state_dict': self.state_dict(),
                                      'embedding_dim': self.embedding_dim,
                                      'hidden_dim': self.hidden_dim,
                                      'layers': self.num_layers}

        torch.save(required_model_information, tmpf.name)
        fname = "trained_LID_model" + fileending + ".pth"
        # exp.add_artifact(tmpf.name, fname)
        tmpf.close()
        os.unlink(tmpf.name)
