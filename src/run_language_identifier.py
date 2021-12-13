
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from lid_model import LIDModel
from language_dataset import LIDDataset, PyTorchLIDDataSet, create_datasplits
from typing import Optional


def test_model(data_set, model: LIDModel):
    model.eval()
    lang_to_idx = model.lang_to_idx
    data_loader = DataLoader(data_set, batch_size=1)
    # 177 is the max number of words in a post
    pred_prob = np.full((len(data_set), 177, 2), -1)

    for i, item in enumerate(tqdm(data_loader, leave=False)):
        words, langs = item
        words = [w[0] for w in words]
        true_labels = [lang[0] for lang in langs]
        preds = model.predict(words)['predictions']
        for word_idx in range(len(words)):
            pred_prob[i, word_idx, 0] = lang_to_idx[preds[word_idx]]
            pred_prob[i, word_idx, 1] = lang_to_idx[true_labels[word_idx]]
    get_stats(lang_to_idx, pred_prob)
    return pred_prob


def get_stats(lang_to_idx, all_preds: np.ndarray):
    conf_mat = np.zeros((len(lang_to_idx), len(lang_to_idx)))
    for i in range(len(all_preds)):
        for pred, gold in all_preds[i]:
            if pred >= 0:
                conf_mat[pred][gold] += 1

    accuracy = conf_mat.trace() / np.sum(conf_mat)
    precision = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    f1 = (2 * precision * recall) / (precision + recall)
    print()
    print("\naccuracy:\t", "\t".join(["{:.4%}".format(accuracy)]))
    print("\t" + "\t".join(list(lang_to_idx.keys())))
    print("precision:\t", "\t".join(["{:.4%}".format(prec) for prec in precision.tolist()]))
    print("recall:\t", "\t".join(["{:.4%}".format(rec) for rec in recall.tolist()]))
    print("f1:\t", "\t".join(["{:.4%}".format(f) for f in f1.tolist()]))


def save_probs(pred_prob, file_ending=""):
    """Saves probabilities as a .npy file and adds it as artifact
    Arguments:
        pred_prob  -- list or numpy array to save as .npy file
    """
    fname = "./prediction_probabilities" + file_ending + ".npy"
    np.save(fname, pred_prob)


def train_model(data_set: PyTorchLIDDataSet, test_dataset: PyTorchLIDDataSet, lidmodel: 'LIDModel',
                training_params, weight_dict: Optional[dict] = None):
    optimizer, weight_decay, lr, batch_size, epochs = training_params
    if optimizer.strip().lower() == "sgd":
        opti = optim.SGD(lidmodel.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opti = optim.AdamW(params=lidmodel.parameters())
    lidmodel.fit(data_set, test_dataset, opti, epochs=epochs, weight_dict=weight_dict, batch_size=batch_size)


def run_training(model, training_params, to_train=True):

    train, dev, test = create_datasplits('./prepped_data/')

    train_dataset = LIDDataset(train)
    dev_dataset = LIDDataset(dev)
    test_dataset = LIDDataset(test)

    train_data_converted = PyTorchLIDDataSet(train_dataset)
    dev_converted = PyTorchLIDDataSet(dev_dataset)

    weight_dict = train_dataset.weight_dict

    if to_train:
        print("Training model")
        train_model(train_data_converted, dev_converted, model,
                    training_params=training_params, weight_dict=weight_dict)

    print("Testing model")
    eval_data = test_model(data_set=test_dataset, model=model)

    print("Saving model")
    model.save_model()

    print("Saving predictions")
    save_probs(eval_data)
