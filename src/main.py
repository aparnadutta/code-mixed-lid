import random
from collections import Counter

from language_dataset import load_posts, write_prep_data, create_datasplits
from torchtext.data.functional import load_sp_model, sentencepiece_tokenizer, sentencepiece_numericalizer
import numpy as np


data2015 = load_posts('./2015data/BN_EN_TRAIN_2015.txt')
fb = load_posts('./data/FB_BN_EN_CR.txt')
twit = load_posts('./data/TWT_BN_EN_CR.txt')
whats = load_posts('./data/WA_BN_EN_CR_CORRECTED.txt')

all_data = data2015 + fb + twit + whats
cleaned = [post for post in all_data if 'bn' in set(post.langs)]
num_tokens = sum([len(post.words) for post in cleaned])
inst_lens = [len(post.words) for post in all_data]
lang_counts = Counter()
for post in all_data:
    lang_counts.update(post.langs)

print("num instances:", len(all_data))
print("num cleaned:", len(cleaned))
print("num tokens:", num_tokens)
print("lang counts:", lang_counts)
print("max num words in post:", np.max(inst_lens))
print("average num words in post:", np.mean(inst_lens))

train, dev, test = [], [], []
random.shuffle(all_data)

ten_perc = int(len(all_data) * 0.1)
train_end = ten_perc * 8
dev_end = train_end + ten_perc

train.extend(all_data[: train_end])
test.extend(all_data[train_end: dev_end])
dev.extend(all_data[dev_end:])


write_prep_data((train, dev, test))


train1, dev1, test1 = create_datasplits('./prepped_data/')
print("len train1:", len(train1))





# gen_sentpiece_model(train)
sp_model = load_sp_model('./spm_user.model')
numericalizer = sentencepiece_numericalizer(sp_model)
tokenizer = sentencepiece_tokenizer(sp_model)
in_words = ['@chairmanwbssc', '@rupakbanerjee10', 'Sir', ',', 'sit', 'e', 'wbcgl', 'mains']
in_string = "@chairmanwbssc @rupakbanerjee10 Sir , sit e wbcgl mains"
output = list(tokenizer(in_words))
id_nums = list(numericalizer(in_words))
# print(output)
# print(id_nums)
