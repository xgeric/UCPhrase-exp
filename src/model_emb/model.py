import utils
import torch
import consts
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from model_base.base import BaseModel
from model_base.base import BaseFeatureExtractor
from utils import move_to


class EmbedModel(BaseModel):
    def __init__(self, model_dir, finetune: bool = False) -> None:
        super().__init__(model_dir=model_dir)
        self.roberta = consts.LM_MODEL
        self.finetune = finetune
        self.dim_len_emb = 50
        self.length_embed = nn.Embedding(num_embeddings=consts.MAX_SUBWORD_GRAM + 1, embedding_dim=self.dim_len_emb)
        self.dim_feature = self.dim_len_emb + consts.LM_MODEL.config.hidden_size * 2
        self.linear_cls_1 = nn.Linear(self.dim_feature, self.dim_feature)
        self.linear_cls_2 = nn.Linear(self.dim_feature, self.dim_feature)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(self.dim_feature, 1)

    @property
    def config(self):
        return {
            'model_dir': str(self.model_dir),
            'finetune': self.finetune,
        }

    @classmethod
    def _from_config(cls, config_dict):
        return cls(
            model_dir=config_dict['model_dir'],
            finetune=config_dict['finetune'])

    def embed_sentences(self, input_ids_batch, input_masks_batch):
        if self.finetune:
            model_output = self.roberta(input_ids_batch, 
                                        attention_mask=input_masks_batch,
                                        output_hidden_states=True,
                                        output_attentions=False,
                                        return_dict=True)
            sentence_embeddings = model_output.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            sentence_embeddings = sentence_embeddings[:, 1:, :]  # remove <s>
        else:
            with torch.no_grad():
                self.roberta.eval()
                model_output = self.roberta(input_ids_batch, 
                                            attention_mask=input_masks_batch,
                                            output_hidden_states=True,
                                            output_attentions=False,
                                            return_dict=True)
            sentence_embeddings = model_output.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            sentence_embeddings = sentence_embeddings[:, 1:, :].detach()  # remove <s>
        return sentence_embeddings

    def forward(self, input_ids_batch, input_masks_batch, spans_batch):
        sentence_embeddings = self.embed_sentences(input_ids_batch, input_masks_batch)
        assert len(input_masks_batch) == len(input_ids_batch) == len(spans_batch) == sentence_embeddings.shape[0]

        span_embs_list = []
        for sent_emb, spans in zip(sentence_embeddings, spans_batch):
            ''' Length Embedding '''
            lens = [[r - l + 1 for l, r in spans]]
            len_idxs = torch.as_tensor(lens, device=consts.DEVICE, dtype=torch.long)
            # ipdb.set_trace()
            len_embs = self.length_embed(len_idxs)[0]
            ''' Token Embeddings '''
            l_idxs = [l for l, r in spans]
            r_idxs = [r for l, r in spans]
            l_embs = sent_emb[l_idxs]
            r_embs = sent_emb[r_idxs]
            ''' Span Embeddings '''
            span_embs = torch.cat([l_embs, r_embs, len_embs], dim=-1)
            assert span_embs.shape == (len(spans), self.dim_feature)
            span_embs_list.append(span_embs)
        span_embs = torch.cat(span_embs_list, dim=0)
        num_spans = sum([len(spans) for spans in spans_batch])
        assert span_embs.shape == (num_spans, self.dim_feature)

        output = self.activation(self.dropout(self.linear_cls_1(span_embs)))
        output = self.activation(self.dropout(self.linear_cls_2(output)))
        logits = self.classifier(output)
        logits = logits.squeeze(-1)
        return logits

    def predict(self, path_tokenized_id_corpus, dir_output, batch_size=128, use_cache=True, max_num_docs=None):
        utils.Log.info(f'Predict: {dir_output}')
        self.eval()

        ''' Decide output path, cachue '''
        dir_output = Path(dir_output)
        path_prefix = f'{max_num_docs}docs.' * (max_num_docs is not None)
        path_output = dir_output / ('pred.' + path_prefix + path_tokenized_id_corpus.name)
        dir_output.mkdir(exist_ok=True)
        if use_cache and utils.IO.is_valid_file(path_output):
            print(f'[ModelPredict] Use cache: {path_output}')
            return path_output

        ''' Predict '''
        tokenized_id_corpus = utils.JsonLine.load(path_tokenized_id_corpus)
        tokenized_id_corpus = tokenized_id_corpus[:max_num_docs] if max_num_docs is not None else tokenized_id_corpus
        marked_sents = [sent for doc in tokenized_id_corpus for sent in doc['sents']]
        sorted_i_sents = sorted(list(enumerate(marked_sents)), key=lambda tup: len(tup[1]['ids']), reverse=True)
        marked_sents = [sent for i, sent in sorted_i_sents]
        sorted_raw_indices = [i for i, sent in sorted_i_sents]
        rawidx2newidx = {rawidx: newidx for newidx, rawidx in enumerate(sorted_raw_indices)}

        batches = BaseFeatureExtractor._batchify(marked_sents, is_train=False)

        with torch.no_grad():
            i_sent = 0
            for input_ids_batch, input_masks_batch, possible_spans_batch in tqdm(batches, ncols=100, desc='Pred'):
                pred_probs = self.get_probs(move_to(input_ids_batch, consts.DEVICE), move_to(input_masks_batch, consts.DEVICE), move_to(possible_spans_batch, consts.DEVICE))
                pred_probs = pred_probs.detach().cpu().numpy()
                assert len(pred_probs) == sum([len(spans) for spans in possible_spans_batch])
                assert input_ids_batch.shape[0] == input_masks_batch.shape[0] == len(possible_spans_batch)
                i_prob = 0
                for possible_spans in possible_spans_batch:
                    marked_sents[i_sent]['spans'] = []
                    for l, r in possible_spans:
                        marked_sents[i_sent]['spans'].append((l, r, pred_probs[i_prob]))
                        i_prob += 1
                    i_sent += 1
        marked_sents = [marked_sents[rawidx2newidx[rawidx]] for rawidx in range(len(marked_sents))]

        pointer = 0
        predicted_docs = []
        num_sents_per_doc = [len(doc['sents']) for doc in tokenized_id_corpus]
        for i_doc, num_sents in enumerate(num_sents_per_doc):
            predicted_docs.append({
                '_id_': tokenized_id_corpus[i_doc]['_id_'],
                'sents': marked_sents[pointer: pointer + num_sents]
            })
            pointer += num_sents
        assert pointer == len(marked_sents)
        assert len(predicted_docs) == len(tokenized_id_corpus)

        utils.Pickle.dump(predicted_docs, path_output)
        return path_output
