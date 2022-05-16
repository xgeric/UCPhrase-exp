import torch

import consts
import utils
import random
from tqdm import tqdm
from consts import DEVICE
from torch.optim import Adam
from model_emb.model import EmbedModel
from model_base.base import BaseFeatureExtractor
from sklearn.metrics import confusion_matrix, f1_score

from utils import move_to


class EmbedTrainer:
    def __init__(self, model: EmbedModel):
        self.model = model.to(DEVICE)
        self.output_dir = model.model_dir
        learning_rate = 1e-5 if self.model.finetune else 1e-3
        print(f'learning rate: {learning_rate}')
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        model_config_path = self.output_dir / 'model_config.json'
        utils.Json.dump(self.model.config, model_config_path)

    def train(self, path_sampled_train_data, num_epochs=20):
        sampled_docs = utils.JsonLine.load(path_sampled_train_data)
        marked_sents = [sent for doc in sampled_docs for sent in doc['sents']]
        marked_sents = sorted(marked_sents, key=lambda s: len(s['ids']), reverse=True)
        batches = BaseFeatureExtractor._batchify(marked_sents, is_train=True)

        num_batches = len(batches)
        num_valid = int(num_batches * 0.1)
        num_train = num_batches - num_valid
        random.shuffle(batches)
        valid_batches = batches[:num_valid]
        train_batches = batches[num_valid:]

        best_epoch = -1
        best_valid_f1 = -1.0
        for epoch in range(1, num_epochs + 1):
            utils.Log.info(f'Epoch [{epoch} / {num_epochs}]')

            ''' Train '''
            epoch_loss = 0.0
            self.model.train()
            random.shuffle(train_batches)
            for input_ids_batch, input_masks_batch, pos_spans_batch, neg_spans_batch in tqdm(train_batches, total=num_train, ncols=100):
                self.model.zero_grad()
                labels = []
                spans_batch = []
                assert len(pos_spans_batch) == len(neg_spans_batch)
                for pos_spans, neg_spans in zip(pos_spans_batch, neg_spans_batch):
                    spans_batch.append(pos_spans + neg_spans)
                    labels += [1] * len(pos_spans) + [0] * len(neg_spans)
                labels = torch.as_tensor(labels, device=DEVICE)
                batch_loss = self.model.get_loss(labels, move_to(input_ids_batch, DEVICE), move_to(input_masks_batch, DEVICE), move_to(spans_batch, DEVICE))
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            train_loss = epoch_loss / num_train
            utils.Log.info(f'Train loss: {train_loss}')

            ''' Valid '''
            self.model.eval()
            gold_labels = []
            pred_labels = []
            with torch.no_grad():
                for input_ids_batch, input_masks_batch, pos_spans_batch, neg_spans_batch in tqdm(valid_batches, total=num_valid, ncols=100):
                    spans_batch = []
                    assert len(pos_spans_batch) == len(neg_spans_batch)
                    for pos_spans, neg_spans in zip(pos_spans_batch, neg_spans_batch):
                        spans_batch.append(pos_spans + neg_spans)
                        gold_labels += [1] * len(pos_spans) + [0] * len(neg_spans)
                    pred_probs = self.model.get_probs(move_to(input_ids_batch, DEVICE), move_to(input_masks_batch, DEVICE), move_to(spans_batch, DEVICE))
                    pred_probs = pred_probs.detach().cpu().numpy().tolist()
                    pred_labels += [int(p > .5) for p in pred_probs]
            valid_f1 = f1_score(gold_labels, pred_labels, average="micro")
            utils.Log.info(f'valid f1: {valid_f1}')
            if valid_f1 < best_valid_f1:
                utils.Log.info(f'Stop training. Best epoch: {epoch - 1}')
                break
            best_epoch = epoch - 1
            best_valid_f1 = valid_f1

            ckpt_dict = {
                'epoch': epoch,
                'model': self.model,
                'valid_f1': valid_f1,
                'train_loss': train_loss,
            }
            ckpt_path = self.output_dir / f'epoch-{epoch}.ckpt'
            torch.save(ckpt_dict, ckpt_path)

        return best_epoch


