
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from lid_model import LIDModel
from datasets import LIDDataset, PyTorchLIDDataSet, Post
from data_loading import get_train_dev_test
from typing import Optional


def test_model(data_set, model: LIDModel) -> tuple[list[Post], np.ndarray]:
    model.eval()
    lang_to_idx = model.lang_to_idx
    data_loader = DataLoader(data_set, batch_size=1)

    conf_mat = np.zeros((len(lang_to_idx), len(lang_to_idx)))
    all_tagged_sents = []

    for i, item in enumerate(tqdm(data_loader, leave=False)):
        tokens, langs = item
        tokens = [w[0] for w in tokens]
        gold_labels = [lang[0] for lang in langs]

        pred_posts = model.predict(tokens)
        pred_labels = pred_posts.langs

        all_tagged_sents.append(pred_posts)

        for pred, gold in zip(pred_labels, gold_labels):
            conf_mat[lang_to_idx[pred]][lang_to_idx[gold]] += 1

    return all_tagged_sents, conf_mat


def save_metrics(conf_mat, lang_labels):
    """Saves metrics as a .txt file
    Arguments:
        conf_mat  -- confusion matrix across dataset
        lang_labels -- list of language labels corresponding with confusion matrix heading
    """
    accuracy = conf_mat.trace() / np.sum(conf_mat)
    precision = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    f1 = (2 * precision * recall) / (precision + recall)

    fname = "./eval_output/test_metrics.txt"
    with open(fname, 'w') as f:
        f.write("\nAccuracy:\t" + "\t".join(["{:.4%}".format(accuracy)]))
        f.write("\nLabels:\t" + "\t".join(lang_labels))
        f.write("\nPrecision:\t" + "\t".join(["{:.4%}".format(prec) for prec in precision.tolist()]))
        f.write("\nRecall:\t" + "\t".join(["{:.4%}".format(rec) for rec in recall.tolist()]))
        f.write("\nF1:\t" + "\t".join(["{:.4%}".format(f) for f in f1.tolist()]))
        f.close()


def save_preds(predictions: list[Post]):
    """Saves predictions as a .txt file
    Arguments:
        preds  -- list of tagged sentences to save
    """
    fname = "./eval_output/test_predictions.txt"
    with open(fname, 'w') as f:
        for post in predictions:
            tagged = ['/'.join((word, tag)) for word, tag in zip(post.words, post.langs)]
            f.write(" ".join(tagged) + "\n")
        f.close()


def train_model(data_set: PyTorchLIDDataSet, test_dataset: PyTorchLIDDataSet, lidmodel: 'LIDModel',
                training_params, weight_dict: Optional[dict] = None):
    optimizer, weight_decay, lr, batch_size, epochs = training_params
    if optimizer.strip().lower() == "sgd":
        opti = optim.SGD(lidmodel.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opti = optim.AdamW(params=lidmodel.parameters())
    lidmodel.fit(data_set, test_dataset, opti, epochs=epochs, weight_dict=weight_dict, batch_size=batch_size)


def run_training(model, training_params, to_train=True, eval_on_test=False):

    train, dev, test = get_train_dev_test('./prepped_data/')

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
    dataset_for_eval = test_dataset if eval_on_test else dev_dataset
    predictions, confusion_mat = test_model(data_set=dataset_for_eval, model=model)

    print("Saving model")
    model.save_model()

    print("Saving predictions")
    save_preds(predictions)
    save_metrics(confusion_mat, list(dataset_for_eval.lang_to_idx.keys()))
