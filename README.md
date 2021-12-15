# nlp-systems-final
Final project for COSI 217a - NLP Systems Fall 2021.

This is a model for word-level language identification for code-mixed Bangla-English social media data, using subword embeddings and a BiLSTM. The file structure, training setup, and LSTM architecture are modified from the code here: https://github.com/AU-DIS/LSTM_langid (this model identifies a single language for a short span of text, while my model identifies a language for each token in the input).

The subword vocabulary is generated using Google SentencePiece, through the torchtext package for PyTorch: https://pytorch.org/text/stable/data_functional.html. The current model is trained using a vocabulary size of 3000, and unigram-based subword embeddings.  


## Data
The data used to train the model is the Bangla-English code-mixed data released from the 2016 and 2015 ICON shared tasks. The original data can be found [here] (http://www.amitavadas.com/Code-Mixing.html). A manually corrected version of the 2016 WhatsApp data is also provided as a part of this project, with 87.5% of the language labels corrected. The corrected WhatsApp data can be found [here](https://github.com/aparnadutta/nlp-systems-final/blob/main/data/WA_BN_EN_CR_CORRECTED.txt).
