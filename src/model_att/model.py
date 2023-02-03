import utils
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from consts import DEVICE
from pathlib import Path
from model_base import BaseModel
from model_att.feature import FeatureExtractor


class AttmapModel(BaseModel):
    def __init__(self, max_num_subwords, num_BERT_layers, model_dir, kernel_size=2, out_channels=16):
        super().__init__(model_dir)
        self.kernel_size = kernel_size
        self.num_BERT_layers = num_BERT_layers
        self.in_channels = num_BERT_layers * 12
        self.out_channels = out_channels
        self.max_num_subwords = max_num_subwords

        self.cnn1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.cnn2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.width_after_pool1 = max_num_subwords - kernel_size + 1
        self.width_after_pool2 = self.width_after_pool1 - kernel_size + 1

        self.cls_linear = nn.Linear(out_channels * self.width_after_pool2 * self.width_after_pool2, 1)

        self.feature_extractor = FeatureExtractor(
            output_dir=self.model_dir.parent / 'features',
            num_BERT_layers=self.num_BERT_layers,
            use_cache=True)

    @property
    def config(self):
        return {
            'max_num_subwords': self.max_num_subwords,
            'num_BERT_layers': self.num_BERT_layers,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'model_dir': str(self.model_dir),
        }

    @classmethod
    def _from_config(cls, config_dict):
        return cls(
            max_num_subwords=config_dict['max_num_subwords'],
            num_BERT_layers=config_dict['num_BERT_layers'],
            model_dir=config_dict['model_dir'],
            kernel_size=config_dict['kernel_size'],
            out_channels=config_dict['out_channels'],
            hidden_size=config_dict['hidden_size'],
        )

    @staticmethod
    def pad_attention_maps(attention_maps: list, max_num_subwords: int):
        num_instances = len(attention_maps)
        num_layers, num_heads = attention_maps[0].shape[:2]
        in_channels = num_layers * num_heads
        padded_tensor = numpy.zeros((num_instances, in_channels, max_num_subwords, max_num_subwords), dtype=numpy.float16)
        for i, attention_map in enumerate(attention_maps):
            width = attention_map.shape[3]
            attention_map = attention_map.reshape(in_channels, width, width)
            padded_tensor[i, :, :width, :width] = attention_map
        return torch.tensor(padded_tensor, dtype=torch.float32)

    def forward(self, attmap_features):
        """
        Args:
            attmap_features: [batch_size, num_layers * num_heads, seqlen, seqlen]
        """
        batch_size, in_channels, seqlen, _ = attmap_features.shape
        assert seqlen == attmap_features.shape[-1] == self.max_num_subwords
        assert in_channels == self.in_channels

        x = attmap_features
        x = F.relu(self.cnn1(x))
        assert x.shape == (batch_size, self.out_channels, self.width_after_pool1, self.width_after_pool1)
        x = F.relu(self.cnn2(x))
        assert x.shape == (batch_size, self.out_channels, self.width_after_pool2, self.width_after_pool2)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        logits = self.cls_linear(x).squeeze(-1)
        return logits

    def predict(self, path_tokenized_id_corpus, dir_output, batch_size=128, use_cache=True, max_num_docs=None):
        utils.Log.info(f'Generate prediction features...')
        path_predict_docs = self.feature_extractor.generate_predict_docs(path_tokenized_id_corpus)

        utils.Log.info(f'Predict: {dir_output}')
        self.eval()

        ''' Decide output path, cachue '''
        dir_output = Path(dir_output)
        path_predict_docs = Path(path_predict_docs)
        path_prefix = f'{max_num_docs}docs.' * (max_num_docs is not None)
        path_output = dir_output / (path_prefix + path_predict_docs.name)
        dir_output.mkdir(exist_ok=True)
        # if use_cache and utils.IO.is_valid_file(path_output):
        #     print(f'[ModelPredict] Use cache: {path_output}')
        #     return path_output
        path_output.parent.mkdir(exist_ok=True)

        predict_docs = utils.Pickle.load(path_predict_docs)
        predict_docs = predict_docs if max_num_docs is None else predict_docs[:max_num_docs]

        predicted_docs = []
        with torch.no_grad():
            for predict_doc in tqdm(predict_docs, ncols=100, desc='predict'):
                predicted_instances = []
                for predict_instance in predict_doc['sents']:
                    spans = predict_instance['spans']
                    spans = [span for span in spans if span[1] - span[0] + 1 <= self.max_num_subwords]
                    spans_batches = utils.get_batches(spans, batch_size)
                    attmap = predict_instance['attmap']
                    predicted_spans = []
                    for batch_spans in spans_batches:
                        span_attmaps = [attmap[:, :, l_idx: r_idx + 1, l_idx: r_idx + 1] for l_idx, r_idx, _ in batch_spans]
                        span_attmaps = self.pad_attention_maps(span_attmaps, max_num_subwords=self.max_num_subwords)
                        span_attmaps = span_attmaps.to(DEVICE)
                        probs = self.get_probs(span_attmaps)
                        probs = probs.detach().cpu().numpy()
                        assert len(probs) == len(batch_spans)
                        for i, (l_idx, r_idx, _) in enumerate(batch_spans):
                            predicted_spans.append((l_idx, r_idx, probs[i]))
                    assert len(predicted_spans) == len(spans)
                    predicted_instance = {
                        'spans': predicted_spans,
                        'ids': predict_instance['ids']
                    }
                    predicted_instances.append(predicted_instance)
                predicted_docs.append({
                    '_id_': predict_doc['_id_'],
                    'sents': predicted_instances})
        utils.Pickle.dump(predicted_docs, path_output)
        return path_output
