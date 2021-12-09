from torchtext.data import sentencepiece_numericalizer, load_sp_model

from src.lstm_model import LSTMLIDModel
from nltk import TreebankWordTokenizer
import torch
from pathlib import Path


class LanguageIdentifier:
    def __init__(self, directory_path: Path):
        model_information_dict = torch.load(directory_path)
        subword_to_idx = sentencepiece_numericalizer(load_sp_model('./spm_user.model'))
        lang_to_idx = {'bn': 0, 'en': 1, 'univ': 2, 'ne': 3, 'hi': 4, 'acro': 5}
        self.model = LSTMLIDModel(subword_to_idx, lang_to_idx,
                                  model_information_dict['embedding_dim'], model_information_dict['hidden_dim'],
                                  model_information_dict['layers'])
        self.model.load_state_dict(model_information_dict['model_state_dict'], strict=False)
        self.tokenizer = TreebankWordTokenizer()

    def tokenize(self, input_sentence: str) -> list[str]:
        return self.tokenizer.tokenize(input_sentence)

    def predict(self, input_sentence: str) -> tuple[str, dict]:
        return input_sentence, self.model.predict(self.tokenize(input_sentence))

    def rank(self, input_sentence: str) -> dict[str, list]:
        return self.model.rank(self.tokenize(input_sentence))


LID = LanguageIdentifier(Path("trained_models/trained_LID_model.pth"))

sent = 'Kakuli Bhattachajeer shaathey dekha kortey jachhi... thesis er kichu topic chhilona discuss kortey ..'

text, preds = LID.predict(sent)

ranking = LID.rank(sent)

# for pred in preds.items():
#     print(pred)
#
# for r in ranking.items():
#     print(r)
