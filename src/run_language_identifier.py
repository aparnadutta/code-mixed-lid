import os
import tempfile

from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from lid_model import LIDModel
from language_dataset import LIDDataset, PyTorchLIDDataSet, create_datasplits
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


def save_probs(pred_prob, file_ending=""):
    """Saves probabilities as a .npy file and adds it as artifact
    Arguments:
        pred_prob  -- list or numpy array to save as .npy file
    """
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmpf.name, pred_prob)
    fname = "prediction_probabilities" + file_ending + ".npy"
    tmpf.close()
    os.unlink(tmpf.name)


def train_model(data_set, test_dataset, lidmodel: 'LIDModel', training_params, weight_dict: Optional[dict] = None):
    optimizer, weight_decay, lr, batch_size, epochs = training_params
    if optimizer.strip().lower() == "sgd":
        opti = optim.SGD(lidmodel.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opti = optim.AdamW(params=lidmodel.parameters())
    lidmodel.fit(data_set, test_dataset, opti, epochs=epochs, weight_dict=weight_dict, batch_size=batch_size)


def run_training(model, training_params, to_train=True):
    data_filepath = './data'
    train, dev, test = create_datasplits(data_filepath)

    train_dataset = LIDDataset(train)
    dev_dataset = LIDDataset(dev)
    test_dataset = LIDDataset(test)

    train_data_converted = PyTorchLIDDataSet(train_dataset)
    dev_converted = PyTorchLIDDataSet(dev_dataset)
    test_data_converted = PyTorchLIDDataSet(test_dataset)

    weight_dict = train_dataset.weight_dict

    if to_train:
        print("Training model")
        train_model(train_data_converted, dev_converted, model,
                    training_params=training_params, weight_dict=weight_dict)

    print("Testing model")
    eval_data = test_model(data_set=test_dataset, model=model)
    print("Saving model")
    model.save_model()
    print("Saving predictions and lang_to_idx")
    save_probs(eval_data)
