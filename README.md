# Word-level LID for Bangla-English Code-mixed Social Media Data
### Final project for COSI 217a - NLP Systems Fall 2021.

This is a model for word-level language identification for code-mixed Bangla-English social media data, using subword embeddings and a BiLSTM. The file structure, training setup, and LSTM architecture are modified from [this](https://github.com/AU-DIS/LSTM_langid) reproduction of Apple's BiLSTM model for short strings. The linked model identifies a single language for a short span of text, while this model identifies a language for each token in the input sentence.

The subword vocabulary is generated using Google SentencePiece, through the [torchtext](https://pytorch.org/text/stable/data_functional.html) package for PyTorch. The current model is trained using a vocabulary size of 3000, and unigram-based subword embeddings.  

## Data
The data used to train the model is the Bangla-English code-mixed data released from the 2016 and 2015 ICON shared tasks. The original data can be found [here](http://www.amitavadas.com/Code-Mixing.html). A manually corrected version of the 2016 WhatsApp data is also provided as a part of this project, with 85.7% of the language labels corrected. The corrected WhatsApp data can be found [here](https://github.com/aparnadutta/nlp-systems-final/blob/main/data/WA_BN_EN_CR_CORRECTED.txt). Note: only the language tags have been modified-- the POS tags have not been checked for errors.


## Example
```python
from LanguageIdentifier import LanguageIdentifier

LID = LanguageIdentifier()
print(LID.predict("amar phone e screenshots er option ache"))
print(LID.rank("amar phone e screenshots er option ache")) 

```
`predict` returns a list of tuples containing each word and its most likely language

`rank` returns a dictionary mapping each language tag to a list of probabilities for each word

```python
[('amar', 'bn'), ('phone', 'en'), ('e', 'bn'), ('screenshots', 'en'), ('er', 'bn'), ('option', 'en'), ('ache', 'bn')]
{'bn': [0.9999899864196777, 7.261116115842015e-05, 0.5536323189735413, 0.0007055602036416531, 0.9999716281890869, 0.4643056392669678, 0.9997987151145935],
'en': [9.930015949066728e-06, 0.999927282333374, 0.44551435112953186, 0.9990161657333374, 9.783412679098547e-06, 0.5293608903884888, 8.509134931955487e-05],
...}

```


###  Files
```
.
├── README.md                     
├── data                          # original data from ICON 2015 and 2016 (with corrected whatsapp data)
├── demo_app                      # streamlit web app with example sentences
├── eval_output                   
│   ├── test_metrics.txt          # accuracy, precision, recall and f1 evaluated on test data
│   └── test_predictions.txt      # test data tagged with predicted labels
|
├── indic-trans                   # submoduled for future use in POS tagging model
├── old_whatsapp_data             # original ICON 2016 whatsapp data before corrections
├── prepped_data                  # shuffled data split into train, dev and test
│   ├── dev.txt
│   ├── test.txt
│   └── train.txt
|
├── requirements.txt              # requirements
├── sentpiece_resources           # trained sentencepiece vocab and model
├── src
│   ├── LanguageIdentifier.py     # class for language identifier
│   ├── data_loading.py           # functions for loading raw files and training the sentencepiece model
│   ├── datasets.py               # Post, Dataset, PyTorchDataset, and BatchSampler classes
│   ├── lid_model.py              # full language ID model and training loop
│   ├── lstm_model.py             # LSTM model that initializes as layers, implements forward method
│   ├── run_training.py           # adjust hyperparameters and train the full model
│   ├── train_test_model.py       # functions for training and testing the model, used in run_training.py
│   └── transliterate_bangla.py   # ex. of indictrans usage for the future-- unimplemented
└── trained_models
    └── trained_lid_model.pth     # the final trained LID model 
```


