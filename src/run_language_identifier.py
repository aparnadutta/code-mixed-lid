from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from lid_model import LIDModel
from language_dataset import LIDDataset, create_datasplits
from typing import Optional


def test_model(data_set, model):
    model.eval()
    lang_to_idx = model.lang_to_idx
    data_loader = DataLoader(data_set, batch_size=1)
    pred_prob = np.zeros((len(data_set), len(lang_to_idx)+1))

    for i, item in enumerate(tqdm(data_loader, leave=False)):
        probs = model.rank(item['text'][0])
        for lang, prob in probs:
            pred_prob[i, lang_to_idx[lang]] = prob
        pred_prob[i, len(lang_to_idx)] = lang_to_idx[item['label'][0]]
    return pred_prob


def idx_maps(path) -> (dict, dict):
    full_dataset = LIDDataset(path)
    lang_to_idx = full_dataset.lang_to_idx
    char_to_idx = full_dataset.char_to_idx
    weight_dict = full_dataset.weight_dict
    return lang_to_idx, char_to_idx, weight_dict


def train_model(exp, data_set, test_dataset, lidmodel: 'LIDModel', training_params, weight_dict: Optional[dict] = None):
    optimizer, weight_decay, lr, batch_size, epochs = training_params
    if optimizer.strip().lower() == "sgd":
        opti = optim.SGD(lidmodel.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opti = optim.AdamW(params=lidmodel.parameters())
    lidmodel.fit(data_set, test_dataset, opti, epochs=epochs, weight_dict=weight_dict,
                 experiment=exp, batch_size=batch_size)


# TODO edit all of this
def run_training(exp, model, maps, training_params, to_train=True):
    # Load train data set

    data_filepath = './data'
    train, dev, test = create_datasplits(data_filepath)
    print("Loading test data")
    test_dataset = load_test_folds()
    lang_to_idx, char_to_idx, weight_dict = maps
    test_dataset.char_to_idx = char_to_idx
    test_dataset.lang_to_idx = lang_to_idx
    test_dataset_converted = PyTorchLIDDataSet(test_dataset)
    if to_train:
        print("Loading train data")
        train_dataset_normal = load_training_folds()
        train_dataset_normal.char_to_idx = char_to_idx
        train_dataset_normal.lang_to_idx = lang_to_idx
        train_dataset = PyTorchLIDDataSet(train_dataset_normal)
        print("Training model")
        train_model(exp, train_dataset, test_dataset_converted, model, weight_dict=weight_dict, training_params=training_params)
    print("Testing model")
    eval_data = test_model(data_set=test_dataset, model=model)
    print("Saving model")
    model.save_model(exp)
    print("Saving predictions and lang_to_idx")
    save_probs(eval_data, exp)
    save_lang_to_idx(test_dataset.lang_to_idx, exp)
