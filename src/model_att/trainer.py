import utils
import torch
import random
from consts import DEVICE
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm
from model_att.model import AttmapModel


class AttmapTrainLoader:
    def __init__(self, random_seed=42, max_num_subwords=10):
        self.random_seed = random_seed
        self.max_num_subwords = max_num_subwords

    def get_batch_size(self):
        return 2048

    def get_loader(self, instances, is_train=True):
        batch_size = self.get_batch_size()
        instances = [instance for instance in instances if instance[2].shape[-1] <= self.max_num_subwords]
        gtlabels = [instance[0] for instance in instances]
        spanlens = [instance[1] for instance in instances]
        attention_maps = [instance[2] for instance in instances]
        gtlabels = torch.tensor(gtlabels, dtype=torch.long)
        spanlens = torch.tensor(spanlens, dtype=torch.long)
        attmap_features = AttmapModel.pad_attention_maps(attention_maps, max_num_subwords=self.max_num_subwords)

        dataset = TensorDataset(attmap_features, gtlabels)
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def load_train_data(self, filepath, sample_ratio=-1):
        print('Loading training data...',)
        instances = utils.Pickle.load(filepath)
        print(f'OK! {len(instances)} training instances')

        if sample_ratio > 0.0:
            assert sample_ratio < 1.0
            num_instances = int(sample_ratio * len(instances))
            instances = random.choices(instances, k=num_instances)
            print(f'[Trainer] Sampled {len(instances)} instances.')

        train_instances, valid_instances = train_test_split(instances, test_size=0.1, shuffle=True, random_state=self.random_seed)
        return self.get_loader(train_instances), self.get_loader(valid_instances)


class AttmapTrainer:
    def __init__(self, model: AttmapModel, sample_ratio=-1):
        self.sample_ratio = sample_ratio

        self.model = model.to(DEVICE)
        self.output_dir = model.model_dir
        self.train_loader = AttmapTrainLoader()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        model_config_path = self.output_dir / 'model_config.json'
        utils.Json.dump(self.model.config, model_config_path)

    def train(self, path_sampled_train_data, num_epochs=20):
        path_train_data = None
        utils.Log.info('Feature extraction...')
        path_train_data = self.model.feature_extractor.generate_train_instances(path_sampled_train_data)

        utils.Log.info(f'Start training: {path_train_data}')
        train_data, valid_data = self.train_loader.load_train_data(path_train_data, sample_ratio=self.sample_ratio)
        num_train = len(train_data)
        num_valid = len(valid_data)

        best_epoch = -1
        best_valid_f1 = -1.0
        for epoch in range(1, num_epochs + 1):
            utils.Log.info(f'Epoch [{epoch} / {num_epochs}]')

            # Train
            self.model.train()
            epoch_loss = 0
            for attmap_features, gtlabels in tqdm(train_data, total=num_train, ncols=100):
                self.model.zero_grad()
                gtlabels = gtlabels.to(DEVICE)
                attmap_features = attmap_features.to(DEVICE)
                batch_loss = self.model.get_loss(gtlabels, attmap_features)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            train_loss = epoch_loss / num_train
            utils.Log.info(f'Train loss: {train_loss}')

            # Valid
            self.model.eval()
            gold_labels = []
            pred_labels = []
            with torch.no_grad():
                for attmap_features, gtlabels in tqdm(valid_data, total=num_valid, ncols=100):
                    attmap_features = attmap_features.to(DEVICE)
                    pred_probs = self.model.get_probs(attmap_features).detach().cpu()
                    gold_labels.extend(gtlabels.numpy().tolist())
                    pred_labels.extend([int(p > .5) for p in pred_probs.numpy().tolist()])
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

