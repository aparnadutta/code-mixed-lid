import random
from collections import Counter

from language_dataset import load_posts, write_prep_data, gen_sentpiece_model, Post, load_file


def load_raw_data() -> list[Post]:
    data2015 = load_posts('./2015data/BN_EN_TRAIN_2015.txt')
    fb = load_posts('./data/FB_BN_EN_CR.txt')
    twit = load_posts('./data/TWT_BN_EN_CR.txt')
    whats = load_posts('./data/WA_BN_EN_CR_CORRECTED.txt')
    all_data = data2015 + fb + twit + whats
    return all_data


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
    train, dev, test = [], [], []

    ten_perc = int(len(all_data) * 0.1)
    train_end = ten_perc * 8
    dev_end = train_end + ten_perc

    train.extend(all_data[: train_end])
    test.extend(all_data[train_end: dev_end])
    dev.extend(all_data[dev_end:])

    write_prep_data(dirpath, (train, dev, test))
    return train, dev, test


def generate_sentencepiece(data_filepath: str = './prepped_data/'):
    train_data = load_file(data_filepath + 'train.txt')
    gen_sentpiece_model(train_data)


def main():
    dirpath = './prepped_data/'
    data = load_raw_data()
    train, dev, test = split_write_data(dirpath, data)
    generate_sentencepiece(dirpath)
    print("\nTotal data stats:")
    print_stats(data)
    print("\nTrain data stats:")
    print_stats(train)
    print("\nDev data stats:")
    print_stats(dev)
    print("\nTest data stats:")
    print_stats(test)


if __name__ == "__main__":
    main()

