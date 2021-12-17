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


def gen_sentpiece_model(vocab_size,
                        model_type: str = 'unigram',
                        train_filepath: str = './prepped_data/train.txt',
                        output_filepath: str = './sp_source_data.txt'):
    """
    Uses training data to generate sentencepiece source data. Creates sentencepiece model of provided
    vocab size and returns the sentencepiece model
    :return: sentencepiece model
    """
    train_data = load_prepped_file(train_filepath)
    with open(output_filepath, 'w') as f:
        for post in train_data:
            f.write(' '.join(post.words) + '\n')
    generate_sp_model(output_filepath, vocab_size=vocab_size, model_prefix='./spm_user', model_type=model_type)


# todo fix this not working
def print_stats(filepaths: list[str]) -> None:
    langs = ['bn', 'en', 'univ', 'ne', 'hi', 'acro', 'mixed', 'undef']

    for f in filepaths:
        dataset: list[Post] = load_raw_posts(f)
        filename = f.rsplit('/', 1)[-1]
        lang_counts = Counter(lang for post in dataset for lang in post)
        num_tokens = sum([len(post) for post in dataset])
        num_utts = str(len(dataset))
        lang_percs = "\t".join(["{:.2%}".format(lang_counts[lang] / num_tokens) for lang in langs])
        print(filename + '\t\t' + str(num_tokens) + num_utts + lang_percs)
        print()


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


def print_cmis(filepaths: list[str]) -> None:
    for f in filepaths:
        dataset: list[Post] = load_raw_posts(f)
        filename = f.rsplit('/', 1)[-1]
        print(filename + '\t\t' + "\t".join(compute_cmi(dataset)))


def compute_cmi(dataset: list[Post]):
    lang_tags = {'bn', 'en', 'hi', 'mixed'}
    non_lang_tags = {'univ', 'acro', 'ne', 'undef'}

    all_cmis = []
    num_tokens = 0

    for post in dataset:
        num_tokens += len(post)
        lang_counts = Counter(lang for lang in post.langs if lang in lang_tags)
        non_lang_counts = Counter(tag for tag in post.langs if tag in non_lang_tags)

        if len(lang_counts) == 0:
            cmi = 0
        else:
            max_wi = lang_counts.most_common(1)[0][1]
            denom = len(post) - sum(non_lang_counts.values())
            cmi = 100 * (1 - (max_wi / denom))
        all_cmis.append(cmi)

    num_tokens = str(num_tokens)
    num_utts = str(len(dataset))
    all_cmi = "{:.4}".format(sum(all_cmis) / len(all_cmis))
    mixed_cmi = "{:.4}".format(sum(all_cmis) / (len(all_cmis) - all_cmis.count(0)))
    code_mix_perc = "{:.2%}".format((len(all_cmis) - all_cmis.count(0)) / len(all_cmis))

    return num_tokens, num_utts, all_cmi, mixed_cmi, code_mix_perc


VOCAB_SIZE = 3000


def main():
    data_sources = ['./data/FB_BN_EN_CR.txt',
                    './data/TWT_BN_EN_CR.txt',
                    './data/WA_BN_EN_CR_CORRECTED.txt',
                    './2015data/BN_EN_TRAIN_2015.txt']
    print_cmis(data_sources)
    print_stats(data_sources)

    # data = [load_raw_posts(f) for f in data_sources]
    # data_dir = './prepped_data/'
    # split_write_data(data_dir, data)
    # gen_sentpiece_model(vocab_size=VOCAB_SIZE, model_type='bpe')


if __name__ == "__main__":
    main()
