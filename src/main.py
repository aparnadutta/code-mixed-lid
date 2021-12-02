from language_dataset import LIDDataset, Post, create_datasplits, gen_sentpiece_model, load_sp_model, load_posts
from torchtext.data.functional import load_sp_model, sentencepiece_tokenizer, sentencepiece_numericalizer


# TODO: called in run_training() in run_language_identifier
# train, dev, test = create_datasplits('./data')

data2015 = load_posts('./2015data/BN_EN_TRAIN_2015.txt')
fb = load_posts('./data/FB_BN_EN_CR.txt')
twit = load_posts('./data/TWT_BN_EN_CR.txt')
whats = load_posts('./data/WA_BN_EN_CR.txt')

all_data = data2015 + fb + twit + whats
cleaned = [post for post in all_data if 'bn' in set(post.langs)]
num_tokens = sum([len(post.words) for post in cleaned])

print("num instances:", len(all_data))
print("num cleaned:", len(cleaned))
print("num tokens:", num_tokens)
# print("words:", [word for post in cleaned for word in post.words])

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
