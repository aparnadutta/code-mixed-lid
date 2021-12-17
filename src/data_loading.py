import random
from collections import Counter
from typing import Optional
import numpy as np
from torchtext.data import generate_sp_model
import re

from src.datasets import Post


def load_raw_posts(filepath: Optional[str]) -> list[Post]:
    """
    Reads the raw data from a file and converts it into a list of Post objects.
    This will later be shuffled and written into three separate files for training, development, and testing.
    :return: one list of Post objects
    """
    all_data = []
    words, langs = [], []

    if filepath is not None:
        with open(filepath) as file:
            for line in file:
                line = line.rstrip()
                if len(line) == 0:
                    if len(words) != 0:
                        all_data.append(Post(words, langs))
                        words, langs = [], []
                else:
                    word, lang, _ = line.split('\t')
                    if '+' in lang:
                        lang = 'mixed'
                    word = re.sub(r'(.)\1{2,}', r"\1", word.lower())
                    words.append(word)
                    langs.append(lang)
    return all_data


def get_train_dev_test(data_filepath: str) -> tuple[list[Post], list[Post], list[Post]]:
    """
    Reads data from the provided directory path and splits the data into train, dev, and test
    :return: 3 lists of posts representing the train data, development data, and test data
    """
    train = load_prepped_file(data_filepath + 'train.txt')
    dev = load_prepped_file(data_filepath + 'dev.txt')
    test = load_prepped_file(data_filepath + 'test.txt')
    return train, dev, test


def write_prep_data(dirpath: str, data: tuple[list[Post], list[Post], list[Post]]) -> None:
    """
    Takes in raw train, dev and test data, and writes to a file with one post per line,
    with tokens and language tags split with a backslash
    :return: 3 lists of posts representing the train data, development data, and test data
    """
    fnames = [f'{dirpath}train.txt', f'{dirpath}dev.txt', f'{dirpath}test.txt']
    for fname, data_chunk in zip(fnames, data):
        with open(fname, 'w') as file:
            for post in data_chunk:
                tagged = ['/'.join((word, tag)) for word, tag in zip(post.words, post.langs)]
                file.write(" ".join(tagged) + '\n')


def load_prepped_file(filepath: str) -> list[Post]:
    """
    Reads the prepared data from one file and converts it into a list of Post objects
    :return: one list of Post objects
    """
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip().split(' ')
            if len(line) > 0:
                words = [pair.rsplit('/', 1)[0] for pair in line]
                langs = [pair.rsplit('/', 1)[1] for pair in line]
                assert len(words) == len(langs)
                data.append(Post(words, langs))
    return data


def gen_sentpiece_model(train_filepath: str = './prepped_data/train.txt',
                        output_filepath: str = './sp_source_data.txt',
                        model_type: str = 'unigram'):
    """
    Uses training data to generate sentencepiece source data. Creates sentencepiece model of provided
    vocab size and returns the sentencepiece model
    :return: sentencepiece model
    """
    train_data = load_prepped_file(train_filepath)
    with open(output_filepath, 'w') as f:
        for post in train_data:
            f.write(' '.join(post.words) + '\n')
    generate_sp_model(output_filepath, vocab_size=VOCAB_SIZE, model_prefix='./spm_user', model_type=model_type)


def load_raw_data() -> list[Post]:
    data_2015 = load_raw_posts('./2015data/BN_EN_TRAIN_2015.txt')
    fb = load_raw_posts('./data/FB_BN_EN_CR.txt')
    twit = load_raw_posts('./data/TWT_BN_EN_CR.txt')
    whats = load_raw_posts('./data/WA_BN_EN_CR_CORRECTED.txt')
    return data_2015 + fb + twit + whats


def print_stats(all_data: list[Post]) -> None:
    num_tokens = sum([len(post.words) for post in all_data])
    inst_lens = [len(post.words) for post in all_data]
    lang_counts = Counter()
    for post in all_data:
        lang_counts.update(post.langs)
    print("num instances:", len(all_data))
    print("num tokens:", num_tokens)
    print("lang counts:", lang_counts)
    for lang, count in lang_counts.items():
        print(lang, "{:.2%}".format(count / num_tokens))
    print("max num words in post:", np.max(inst_lens))
    print("average num words in post:", np.mean(inst_lens))


def split_write_data(dirpath, all_data: list[Post]) -> tuple[list[Post], list[Post], list[Post]]:
    random.shuffle(all_data)
    # 60:20:20 split
    twenty_perc = int(len(all_data) * 0.2)
    train_end = twenty_perc * 3
    test_end = train_end + twenty_perc

    train = all_data[: train_end]
    test = all_data[train_end: test_end]
    dev = all_data[test_end:]

    write_prep_data(dirpath, (train, dev, test))
    return train, dev, test


def compute_cmi(dataset: list[Post]) -> tuple[float, float]:
    all_cmis = []
    for post in dataset:
        lang_counts = Counter(post.langs)
        max_wi = lang_counts.most_common(1)[0][1]
        n = len(post.langs)
        u = sum([count for lang, count in lang_counts.items() if lang in {'univ', 'acro', 'undef'}])
        if n == u:
            cmi = 0
        else:
            cmi = 100 * (1 - (max_wi / (n - u)))
        all_cmis.append(cmi)
    all_cmi = sum(all_cmis) / len(all_cmis)
    mixed_cmi = sum(all_cmis) / (len(all_cmis) - all_cmis.count(0))
    print("all cmi:", all_cmi)
    print("mixed cmi:", mixed_cmi)
    print("num words:", sum([len(post.words) for post in dataset]))
    print("num utterances:", len(dataset))
    print("perc mixed:", (len(all_cmis) - all_cmis.count(0)) / len(all_cmis))
    return all_cmi, mixed_cmi


VOCAB_SIZE = 3000


def main():
    # data_dir = './prepped_data/'
    # data = load_raw_data()
    # compute_cmi(data)
    # split_write_data(data_dir, data)
    gen_sentpiece_model(model_type='bpe')


if __name__ == "__main__":
    main()


