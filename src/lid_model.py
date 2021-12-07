import os
import tempfile
import time
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from language_dataset import BatchSampler, VOCAB_SIZE, PyTorchLIDDataSet


def correct_predictions(scores, masks, labels):
    pred = torch.argmax(scores, dim=1)

    masked_pred = torch.masked_select(pred, masks)
    masked_labels = torch.masked_select(labels, masks)

    num_corr = masked_pred == masked_labels
    num_corr = num_corr.sum()
    return num_corr


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


class LIDModel(nn.Module):
    def __init__(self, subword_to_idx, lang_to_idx):
        # Subword_to_idx is a function that converts a subword to a number, and converts unknown tokens to 0
        # Lang_to_idx is a map that converts a language to a number
        self.subword_to_idx = subword_to_idx
        self.lang_to_idx = lang_to_idx
        self.idx_to_lang = dict([(value, key) for key, value in lang_to_idx.items()])
        self.vocab_size = VOCAB_SIZE
        self.lang_set_size = len(lang_to_idx)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(LIDModel, self).__init__()

    def pad_collate(self, batch):
        (words, masks, labels) = batch[0]
        words = words.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)
        return words, masks, labels

    def prepare_sequence(self, sentence):
        word_id = list(self.subword_to_idx(sentence))
        mask_nest = [[True] + [False] * (len(word_id[num]) - 1) for num in range(len(word_id))]

        word_ids_flat = [w_id for word in word_id for w_id in word]
        mask_flat = [idx for word in mask_nest for idx in word]

        id_tensor = torch.tensor(word_ids_flat, dtype=torch.long, device=self.device).view(1, len(word_ids_flat))
        mask_tensor = torch.tensor(mask_flat, dtype=torch.bool, device=self.device)

        return id_tensor, mask_tensor

    def forward(self, sentence):
        raise NotImplemented

    def predict(self, sentence):
        self.eval()
        prep_sent, mask = self.prepare_sequence(sentence)

        feats = self(prep_sent).transpose(1, 2)  # shape (batch_size, seq_len, num_labels)
        feats_smax = F.log_softmax(feats, dim=-1).squeeze()  # log-softmaxing over each word

        preds = torch.argmax(feats_smax, dim=-1)
        masked_preds = torch.masked_select(preds, mask)

        lang_preds = [self.idx_to_lang[pred.item()] for pred in masked_preds]

        self.train()
        return lang_preds

    def rank(self, sentence):
        self.eval()
        prep_sent, mask = self.prepare_sequence(sentence)
        feats = self(prep_sent).transpose(1, 2)  # shape (batch_size, seq_len, num_labels)
        feats_smax = F.log_softmax(feats, dim=-1).squeeze()  # log-softmaxing over each word

        num_labels = feats_smax.size()[-1]
        masked_feats = torch.masked_select(feats_smax.T, mask)
        masked_feats = masked_feats.reshape(len(sentence), num_labels)

        arr = []

        for word_i in range(len(sentence)):
            word_rank = []
            for lang, lang_idx in self.lang_to_idx.items():
                word_rank.append((lang, masked_feats[word_i][lang_idx].item()))
            arr.append(word_rank)
        self.train()
        return arr

    def fit(self, train_dataset: PyTorchLIDDataSet, dev_dataset: PyTorchLIDDataSet,
            optimizer, epochs=3, batch_size=64, weight_dict=None):
        test_sampler = BatchSampler(batch_size, dev_dataset)
        dataloader_dev = DataLoader(dev_dataset, shuffle=False, drop_last=False, collate_fn=self.pad_collate, sampler=test_sampler)

        weights = None
        if weight_dict is not None:
            weights = torch.zeros(len(weight_dict)).to(self.device)
            for lang in weight_dict:
                indx = self.lang_to_idx[lang]
                weights[indx] = weight_dict[lang]

        # Index of the dummy label is -1
        loss_train = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
        loss_dev = nn.CrossEntropyLoss(ignore_index=-1)

        print(f"Running for {epochs} epochs")
        for epoch in range(epochs):
            self.train()
            avg_total_loss, num_correct_preds = 0, 0
            epoch_start_time = time.time()

            sampler = BatchSampler(batch_size, train_dataset)
            dataloader_train = DataLoader(train_dataset, shuffle=False, drop_last=False, collate_fn=self.pad_collate, sampler=sampler)

            # Logit is the pre-softmax scores
            for idx, batch in enumerate(tqdm(dataloader_train, leave=False)):
                optimizer.zero_grad()
                tensor_sentences, masks, labels = batch
                logit = self(tensor_sentences)

                loss_nll = loss_train(logit, labels)
                num_correct_preds += correct_predictions(logit, masks, labels)
                loss = loss_nll
                avg_total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_total_loss /= sampler.batch_count()

            train_num_tokens = sum(train_dataset.all_post_lens)
            accuracy = (num_correct_preds / train_num_tokens).item()

            print(f"\nAverage training error in epoch {epoch + 1}: {avg_total_loss:.5f} "
                      f"and training accuracy: {accuracy:.4f}")
            step_num = epoch
            print("Training Accuracy:", accuracy, step_num)
            print("Training Loss:", avg_total_loss, step_num)
            self.eval()
            # Test model
            avg_total_loss, num_correct_preds = 0, 0
            for _, batch in enumerate(tqdm(dataloader_dev, leave=False)):
                tensor_sentences, masks, labels = batch
                logit = self(tensor_sentences)
                loss_nll = loss_dev(logit, labels)
                num_correct_preds += correct_predictions(logit, masks, labels)
                avg_total_loss += loss_nll.item()
            avg_total_loss /= test_sampler.batch_count()

            dev_num_tokens = sum(dev_dataset.all_post_lens)
            accuracy = (num_correct_preds / dev_num_tokens).item()

            print(f"\nAverage total loss dev: {avg_total_loss:.5f}, accuracy: {accuracy:.4f}, ")
            print("Dev Accuracy:", accuracy, step_num)
            print("Dev Loss:", avg_total_loss, step_num)
            # self.save_model("E" + str(epoch))
            print("Time spent in epoch {0}: {1:.2f} ".format(epoch + 1, time.time() - epoch_start_time))

    def save_model(self, fileending=""):
        """Saves a pytorch model fully
        """
        required_model_information = {'model_state_dict': self.state_dict()}
        fname = "./trained_model_dict" + fileending + ".pth"
        torch.save(required_model_information, fname)
