import math
import os
import tempfile
import time
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from language_dataset import BatchSampler


def correct_predictions(scores, labels):
    pred = torch.argmax(scores, dim=1)
    return (pred == labels).float().sum()


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


class LIDModel(nn.Module):
    def __init__(self, subword_to_idx, lang_to_idx):
        # Subword_to_idx should be a map that converts a subword to a number
        # Lang_to_idx should be a map that converts a language to a number
        self.subword_to_idx = subword_to_idx
        self.lang_to_idx = lang_to_idx
        self.idx_to_lang = dict([(value, key) for key, value in lang_to_idx.items()])
        self.vocab_size = len(subword_to_idx)
        self.lang_set_size = len(lang_to_idx)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(LIDModel, self).__init__()

    def pad_collate(self, batch):
        (words, labels) = batch[0]
        words = words.to(self.device)
        labels = labels.to(self.device)
        return words, labels

    def prepare_sequence(self, sentence):
        idxs = [self.subword_to_idx(sub) for sub in sentence]
        return torch.tensor(idxs, dtype=torch.long, device=self.device).view(1, len(sentence))

    def forward(self, sentences):
        raise NotImplemented

    def predict(self, sentence):
        self.eval()
        prep_sent = self.prepare_sequence(sentence)
        feats = F.log_softmax(self(prep_sent), dim=-1)
        lang_preds = [self.idx_to_lang[argmax(word[0])] for word in feats]
        self.train()
        return lang_preds

    def rank(self, sentence):
        self.eval()
        prep_sentence = self.prepare_sequence(sentence)
        logit = self(prep_sentence)
        smax = F.log_softmax(logit, dim=-1)
        arr = []
        for lang, index in self.lang_to_idx.items():
            # TODO i think this might cause a problem, had to index into predict above
            arr.append((lang, smax[0][index].item()))
        self.train()
        return arr

    def fit(self, train_dataset, dev_dataset, optimizer, epochs=3, batch_size=64, weight_dict=None):
        test_sampler = BatchSampler(batch_size, dev_dataset)
        dataloader_dev = DataLoader(dev_dataset, shuffle=False, drop_last=False, collate_fn=self.pad_collate, sampler=test_sampler)
        weights = None
        if weight_dict is not None:
            weights = torch.zeros(len(weight_dict)).to(self.device)
            for lang in weight_dict:
                indx = self.lang_to_idx[lang]
                weights[indx] = weight_dict[lang]
        loss_train = nn.CrossEntropyLoss(weight=weights)
        loss_dev = nn.CrossEntropyLoss()

        print(f"Running for {epochs} epochs")
        for epoch in range(epochs):
            self.train()
            avg_total_loss, num_correct_preds = 0, 0
            epoch_start_time = time.time()
            train_dataset.randomize_data()
            sampler = BatchSampler(batch_size, train_dataset)
            dataloader_train = DataLoader(train_dataset, shuffle=False, drop_last=False, collate_fn=self.pad_collate, sampler=sampler)
            # Logit is the pre-softmax scores
            for idx, batch in enumerate(tqdm(dataloader_train, leave=False)):
                optimizer.zero_grad()
                tensor_sentences, labels = batch
                logit = self(tensor_sentences)

                # Todo need to modify the label set so that it's the same length as BPE sentence
                loss_nll = loss_train(logit, labels)
                num_correct_preds += correct_predictions(logit, labels)
                loss = loss_nll
                avg_total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_total_loss /= sampler.batch_count()
            accuracy = num_correct_preds / len(train_dataset)
            print(f"Average training error in epoch {epoch + 1}: {avg_total_loss:.5f} "
                      f"and training accuracy: {accuracy:.4f}")
            step_num = epoch
            print("Training Accuracy:", accuracy, step_num)
            print("Training Loss:", avg_total_loss, step_num)
            self.eval()
            # Test model
            avg_total_loss, num_correct_preds = 0, 0
            for _, batch in enumerate(tqdm(dataloader_dev, leave=False)):
                tensor_sentences, labels = batch
                logit = self(tensor_sentences)
                loss_nll = loss_dev(logit, labels)
                num_correct_preds += correct_predictions(logit, labels)
                avg_total_loss += loss_nll.item()
            avg_total_loss /= test_sampler.batch_count()
            accuracy = num_correct_preds / len(dev_dataset)
            print(f"Average total loss dev: {avg_total_loss:.5f}, accuracy: {accuracy:.4f}, ")
            print("Dev Accuracy:", accuracy, step_num)
            print("Dev Loss:", avg_total_loss, step_num)
            self.save_model("E" + str(epoch))
            print("Time spent in epoch {0}: {1:.2f} ".format(epoch + 1, time.time() - epoch_start_time))

    def save_model(self, fileending=""):
        """Saves a pytorch model fully and adds it as artifact
        Arguments:
            pred_prob  -- list or numpy array to save as .npy file
        """
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        required_model_information = {'subword_to_idx': self.subword_to_idx, 'lang_to_idx': self.lang_to_idx,
                                      'model_state_dict': self.state_dict()}

        torch.save(required_model_information, tmpf.name)
        fname = "trained_model_dict" + fileending + ".pth"
        # exp.add_artifact(tmpf.name, fname)
        tmpf.close()
        os.unlink(tmpf.name)
