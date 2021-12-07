from torchtext.data import sentencepiece_numericalizer, load_sp_model

# Use uvicorn to run this:
# uvicorn src.LanguageIdentifier:app --reload
from lstm_model import LSTMLIDModel
import torch
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class LanguageIdentifier:
    def __init__(self, directory_path: Path):
        model_information_dict = torch.load(directory_path)
        subword_to_idx = sentencepiece_numericalizer(load_sp_model('../spm_user.model'))
        lang_to_idx = {'bn': 0, 'en': 1, 'univ': 2, 'ne': 3, 'hi': 4, 'acro': 5}
        self.model = LSTMLIDModel(subword_to_idx, lang_to_idx,
                                  model_information_dict['embedding_dim'], model_information_dict['hidden_dim'],
                                  model_information_dict['layers'])
        self.model.load_state_dict(model_information_dict['model_state_dict'], strict=False)

    def tokenize(self, text: str) -> list[str]:
        return text.split(" ")

    def predict(self, text: str) -> tuple[str, list[str]]:
        return text, self.model.predict(self.tokenize(text))


lid = LanguageIdentifier(Path("../trained_LID_model.pth"))
print(lid.predict("amar phone e screen shots er option ache"))
print(lid.predict("oder car ta drive kore school e jacchi"))


# To handle POST, we need to declare a schema for the body. BaseModel from pydantic works
# much like attrs(auto_attribs=true) or inheriting from NamedTuple
class LIDRequest(BaseModel):
    text: str


class LIDResponse(BaseModel):
    text: str
    tokens: list[str]
    tags: list[str]


# GET using query parameters (i.e. ?text=foo) is very easy, just declare it in the function
# Note that functions are declared with `async def` by default here
@app.get("LID")
async def lid_tag_get(text: str) -> LIDResponse:
    return LIDResponse(text=text, tokens=lid.tokenize(text), tags=lid.predict(text))


@app.post("LID")
async def lid_tag_post(body: LIDRequest) -> LIDResponse:
    text = body.text
    return LIDResponse(text=text, tokens=lid.tokenize(text), tags=lid.predict(text))
