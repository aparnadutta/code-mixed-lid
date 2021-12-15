import random
from collections import Counter

import numpy as np

from language_dataset import load_posts, write_prep_data, gen_sentpiece_model, Post, load_file


def load_raw_data() -> list[Post]:
    data_2015 = load_posts('./2015data/BN_EN_TRAIN_2015.txt')
    fb = load_posts('./data/FB_BN_EN_CR.txt')
    twit = load_posts('./data/TWT_BN_EN_CR.txt')
    whats = load_posts('./data/WA_BN_EN_CR_CORRECTED.txt')
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


def generate_sentencepiece(data_filepath='./prepped_data/'):
    train_data = load_file(data_filepath + 'train.txt')
    gen_sentpiece_model(train_data)


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


def main():
    dirpath = './prepped_data/'
    # data = load_raw_data()
    # compute_cmi(data)
    # train, dev, test = split_write_data(dirpath, data)
    generate_sentencepiece(dirpath)
    # print("\nTotal data stats:")
    # print_stats(data)
    # print("\nTrain data stats:")
    # print_stats(train)
    # print("\nDev data stats:")
    # print_stats(dev)
    # print("\nTest data stats:")
    # print_stats(test)


if __name__ == "__main__":
    main()


