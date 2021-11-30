from language_dataset import LIDDataset, Post, create_datasplits, gen_sentpiece_model, load_sp_model
from torchtext.data.functional import load_sp_model, sentencepiece_tokenizer, sentencepiece_numericalizer


# TODO: called in run_training() in run_language_identifier
# train, dev, test = create_datasplits('../data')
# gen_sentpiece_model(train)
sp_model = load_sp_model('./spm_user.model')
numericalizer = sentencepiece_numericalizer(sp_model)
tokenizer = sentencepiece_tokenizer(sp_model)
in_words = ['@chairmanwbssc', '@rupakbanerjee10', 'Sir', ',', 'sit', 'e', 'wbcgl', 'mains']
in_string = "@chairmanwbssc @rupakbanerjee10 Sir , sit e wbcgl mains"
output = list(tokenizer(in_words))
id_nums = list(numericalizer(in_words))
print(output)
print(id_nums)
