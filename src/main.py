from language_dataset import LIDDataset, Post, create_datasplits, gen_sentpiece_model, load_sp_model
# from indictrans import Transliterator
from torchtext.data.functional import load_sp_model, sentencepiece_tokenizer, sentencepiece_numericalizer

# trn = Transliterator(source='eng', target='ben', build_lookup=True)

# TODO: called in run_training() in run_language_identifier
# train, dev, test = create_datasplits('../data')
# gen_sentpiece_model(train)
sp_model = load_sp_model('./spm_user.model')
numericalizer = sentencepiece_numericalizer(sp_model)
tokenizer = sentencepiece_tokenizer(sp_model)
in_words = ['amader', 'toder', 'college', 'e']
output = list(tokenizer(in_words))
id_nums = list(numericalizer(in_words))
print(output)
print(id_nums)


# train_dataset = LIDDataset(train, sp_model)


# weights = train_dataset.weight_dict
# print("weights:", weights)
#
# test_item_raw: Post = train_dataset.data[0]
# test_item_2: Post = train_dataset.subword_data[0]
#
# print("test item words:", test_item_raw.words)
# print("test_item_2 words:", test_item_2.words)


# def span_identifier(data: Post) -> dict[str, list]:
#     spans = {lang: [] for lang in data.langs}
#     cur_list = [0]
#     for i in range(1, len(data)):
#         last_lang_idx = cur_list[-1]
#         if data.langs[i] == data.langs[last_lang_idx]:
#             cur_list.append(i)
#         else:
#             spans[data.langs[i]].append(cur_list)
#             cur_list = [i]
#     return spans
#
#
# data_spans = span_identifier(test_item)
# print(data_spans)
