# Word-level LID for Bangla-English Code-mixed Social Media Data
### Final project for COSI 217a - NLP Systems Fall 2021.

This is a model for word-level language identification for code-mixed Bangla-English social media data, using subword embeddings and a BiLSTM. The file structure, training setup, and LSTM architecture are modified from [this](https://github.com/AU-DIS/LSTM_langid) reproduction of Apple's BiLSTM model for short strings. The linked model identifies a single language for a short span of text, while this model identifies a language for each token in the input sentence.

The subword vocabulary is generated using Google SentencePiece, through the [torchtext](https://pytorch.org/text/stable/data_functional.html) package for PyTorch. The current model is trained using a vocabulary size of 3000, and unigram-based subword embeddings.  

## Data
The data used to train the model is the Bangla-English code-mixed data released from the 2016 and 2015 ICON shared tasks. The original data can be found [here](http://www.amitavadas.com/Code-Mixing.html). A manually corrected version of the 2016 WhatsApp data is also provided as a part of this project, with 87.5% of the language labels corrected. The corrected WhatsApp data can be found [here](https://github.com/aparnadutta/nlp-systems-final/blob/main/data/WA_BN_EN_CR_CORRECTED.txt). Note: only the language tags have been modified-- the POS tags have not been checked for errors.


## Example
```python
from LanguageIdentifier import LanguageIdentifier

LID = LanguageIdentifier()
print(predict("amar phone e screen shots er option ache"))
print(rank("This is english")) 
```
`predict` returns a list of tuples containing each word and its most likely language. 
`rank` returns ...


###  Files
```
.
├── README.md                     
├── data                          #
├── demo_app                      #
├── eval_output                   #
│   ├── test_metrics.txt
│   └── test_predictions.txt
|
├── indic-trans
├── old_whatsapp_data             #
├── prepped_data                  #
│   ├── dev.txt
│   ├── test.txt
│   └── train.txt
|
├── requirements.txt              #
├── sentpiece_resources           #
├── src
│   ├── LanguageIdentifier.py     # 
│   ├── data_loading.py           #
│   ├── datasets.py               #
│   ├── lid_model.py              #
│   ├── lstm_model.py             #
│   ├── run_training.py           #
│   ├── train_test_model.py       #
│   └── transliterate_bangla.py   #
└── trained_models
```


