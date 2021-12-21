#!/usr/bin/env python

from collections import Counter
from typing import Callable

import torch
from torch.utils.data import Sampler, Dataset
from torchtext.data.functional import load_sp_model, sentencepiece_numericalizer
from random import shuffle


class Post:
    """A single social media data instance consisting of
    a list of word tokens and a list of language tags"""

    def __init__(self, words: list[str], langs: list[str]):
        self.words = words
        self.langs = langs

    def __getitem__(self, idx):
        return self.words[idx], self.langs[idx]

    def __len__(self):
        return len(self.words)


class LIDDataset(Dataset):
    def __init__(self, dataset):
        self.data: list[Post] = dataset
        self.sp_model = load_sp_model('./sentpiece_resources/spm_user.model')
        self.subword_to_idx: Callable = sentencepiece_numericalizer(self.sp_model)
        self.lang_to_idx: dict[str, int] = {'bn': 0, 'en': 1, 'univ': 2,
                                            'ne': 3, 'hi': 4, 'acro': 5, 'mixed': 6, 'undef': 7}
        self.weight_dict = self.make_weight_dict()

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
            weight_dict = {label: (most_frequent / frequency_dict[label]) if frequency_dict[label] != 0 else 0
                           for label in frequency_dict.keys()}
        return weight_dict

    def __getitem__(self, idx) -> tuple[list[str], list[str]]:
        idx_item = self.data[idx]
        assert len(idx_item.words) == len(idx_item.langs)
        return idx_item.words, idx_item.langs

    def __len__(self):
        return len(self.data)

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


class PyTorchLIDDataSet(Dataset):
    """
    PyTorch-specific wrapper that converts items to PyTorch tensors.
    """

    def __init__(self, decoree: LIDDataset):
        self.data = []
        self.all_post_lens = []
        if decoree is not None:
            self.decoree = decoree
        self.subword_to_idx = decoree.subword_to_idx
        self.lang_to_idx = decoree.lang_to_idx
        self.tensorify_all()

    def __getitem__(self, idx):
        if not isinstance(idx, list):
            return self.data[idx]
        txt = []
        mask = []
        label = []
        for i in idx:
            item = self.data[i]
            txt.append(item[0])
            mask.append(item[1])
            label.append(item[2])

        txts = torch.stack(txt)
        masks = torch.stack(mask)
        labels = torch.stack(label)

        return txts, masks, labels

    def make_weight_dict(self) -> dict:
        return self.decoree.make_weight_dict()

    def __len__(self):
        return len(self.data)

    def get_tag_set(self) -> list:
        return self.decoree.get_tag_set()

    def get_lang_to_idx(self) -> dict:
        """get dict from lang to id, ordered alphabetically
        Returns:
            dict -- For converting language code to an id
        """
        return self.decoree.get_lang_to_idx()

    def tensorify(self, data_point: tuple[list[str], list[str]]):
        words, langs = data_point
        self.all_post_lens.append(len(words))
        word_id = list(self.subword_to_idx(words))
        lang_id = [self.lang_to_idx[lang] for lang in langs]

        # The first subword is assigned the true label, all other subwords are assigned the dummy label -1
        lang_id_pad = [[lang_id[word_num]] + [-1] * (len(word_id[word_num]) - 1) for word_num in range(len(word_id))]

        word_ids_flat = [w_id for word in word_id for w_id in word]
        lang_ids_flat = [l_id for lang in lang_id_pad for l_id in lang]

        mask_nest = [[True] + [False] * (len(word_id[num]) - 1) for num in range(len(word_id))]
        mask = [idx for word in mask_nest for idx in word]

        return torch.tensor(word_ids_flat, dtype=torch.long), \
               torch.tensor(mask, dtype=torch.bool), \
               torch.tensor(lang_ids_flat, dtype=torch.long)

    def tensorify_all(self):
        new_data = []
        for elem in self.decoree:
            new_data.append(self.tensorify(elem))
        self.data = new_data

    def set_lang_to_idx(self, l_to_idx):
        self.lang_to_idx = l_to_idx
        self.decoree.lang_to_idx = l_to_idx

    def set_subword_to_idx(self, s_to_idx):
        self.subword_to_idx = s_to_idx
        self.decoree.char_to_idx = s_to_idx


# ----------------------------------------------------------------------------------------------------------------------
# Based on https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/utils/data/text/dataset.py
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
