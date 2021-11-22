#!/usr/bin/env python
import random
import os
from collections import Counter
from typing import Optional, List, Dict, Tuple

from torch.utils.data import Sampler, Dataset
from torchtext.data.functional \
    import generate_sp_model, load_sp_model, sentencepiece_tokenizer, sentencepiece_numericalizer
from random import shuffle


class Post:
    """A single social media data instance"""

    def __init__(self, words: List[str], langs: List[str]):
        self.words = words
        self.langs = langs

    def __getitem__(self, idx):
        return self.words[idx], self.langs[idx]

    def __len__(self):
        return len(self.words)


def gen_sentpiece_model(training_data: List[Post]):
    sp_filepath = './sp_source_data.txt'
    with open(sp_filepath, 'a') as f:
        for post in training_data:
            f.write(' '.join(post.words) + '\n')
    generate_sp_model(sp_filepath, vocab_size=1000, model_prefix='./spm_user')
    sp_model = load_sp_model('./spm_user.model')
    return sp_model


def create_datasplits(data_filepath: str) -> Tuple[List[Post], List[Post], List[Post]]:
    files = [load_posts(f'{data_filepath}/{file}') for file in os.listdir(data_filepath)]

    train, dev, test = [], [], []
    for f in files:
        ten_perc = int(len(f) * 0.1)
        train_end = ten_perc*8
        dev_end = train_end + ten_perc

        train.extend(f[: train_end])
        test.extend(f[train_end: dev_end])
        dev.extend(f[dev_end:])

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test


def load_posts(filepath: Optional[str]) -> List[Post]:
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
                    word, lang = line.split('\t')[:-1]
                    words.append(word)
                    langs.append(lang)
    return all_data


class LIDDataset(Dataset):
    def __init__(self, dataset):
        self.data: List[Post] = dataset
        self.subword_data: List[Post] = []
        self.sp_tokenizer = sentencepiece_tokenizer(load_sp_model('./spm_user.model'))
        self.subword_to_idx: Dict[str, int] = sentencepiece_numericalizer(load_sp_model('./spm_user.model'))
        self.lang_to_idx: Dict[str, int] = {'bn': 0, 'univ': 1, 'en+bn_suffix': 2, 'undef': 3,
                                            'hi': 4, 'ne': 5, 'en': 6, 'acro': 7, 'ne+bn_suffix': 8}
        self.weight_dict = self.make_weight_dict()

        sub_data = []
        for post in self.data:
            new_words = list(self.sp_tokenizer(post.words))
            sub_data.append(Post(new_words, post.langs))
        self.subword_data = sub_data

    def make_weight_dict(self) -> dict:
        """
        Instantiates the weight dict for this dataset
        The formula used is weight = most_frequent/lang_freq.
        Such that the most frequent has a frequency of 1
        :return: A dict with a mapping from a language to weight
        """
        weight_dict = None
        if len(self.data) > 0:
            frequency_dict = {}
            label_counts = Counter([lang for sentence in self.data for lang in sentence.langs])
            for label in self.lang_to_idx.keys():
                frequency_dict[label] = label_counts[label]
            most_frequent = max(frequency_dict.values())
            weight_dict = {label: (most_frequent / frequency_dict[label]) for label in frequency_dict.keys()}
        return weight_dict

    def __getitem__(self, idx) -> Post:
        return self.subword_data[idx]

    def __len__(self):
        return len(self.subword_data)

    def get_tag_set(self) -> list:
        """returns ordered list of language labels in the dataset
        Returns:
            list -- Ordered list of language labels
        """
        langs = list(self.lang_to_idx.keys())
        langs.sort()
        return langs

    def get_lang_to_idx(self) -> dict:
        """get dict from lang to id, ordered alphabetically
        Returns:
            dict -- For converting language code to an id
        """
        lang_to_idx = {}
        for lang in self.get_tag_set():
            lang_to_idx[lang] = len(lang_to_idx)
        return lang_to_idx


class BatchSampler(Sampler):
    """
    This class creates batches containing equal length examples.
    """

    def __init__(self, batch_size, inputs):
        self.batch_size = batch_size
        self.input = inputs
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        batch_map = {}
        for idx, item in enumerate(self.input):
            length = len(item[0])
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
