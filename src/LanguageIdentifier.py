from torchtext.data import sentencepiece_numericalizer, load_sp_model, sentencepiece_tokenizer

from lstm_model import LSTMLIDModel
from nltk import TreebankWordTokenizer
import torch
from pathlib import Path


class LanguageIdentifier:
    """
    Word-level language identifier
    """
    def __init__(self, directory_path: Path = "./trained_models/trained_LID_model.pth"):
        model_information_dict = torch.load(directory_path)
        subword_to_idx = sentencepiece_numericalizer(load_sp_model('./sentpiece_resources/spm_user.model'))
        lang_to_idx = {'bn': 0, 'en': 1, 'univ': 2, 'ne': 3, 'hi': 4, 'acro': 5, 'mixed': 6, 'undef': 7}
        self.model = LSTMLIDModel(subword_to_idx, lang_to_idx,
                                  model_information_dict['embedding_dim'],
                                  model_information_dict['hidden_dim'],
                                  model_information_dict['layers'])
        self.model.load_state_dict(model_information_dict['model_state_dict'], strict=False)
        self.tokenizer = TreebankWordTokenizer()

    def tokenize(self, input_sentence: str) -> list[str]:
        return self.tokenizer.tokenize(input_sentence)

    def predict(self, input_sentence: str) -> list[tuple[str, str]]:
        tokens = self.tokenize(input_sentence)
        return self.model.predict(tokens)

    def rank(self, input_sentence: str) -> dict[str, list]:
        tokens = self.tokenize(input_sentence)
        return self.model.rank(tokens)


# To instantiate an instance of the language identifier
# LID = LanguageIdentifier()
# ex_sent = "ami oke plant ta dekhiyechi"
# print(LID.predict(ex_sent))

