import utils
import string
import consts
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

PUNCS = set(string.punctuation) - {'-'}


class BaseModel(nn.Module):
    def __init__(self, model_dir) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(.2)
        self.loss = nn.BCEWithLogitsLoss()

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

    @property
    def config(self):
        raise NotImplementedError

    @classmethod
    def _from_config(cls, config_dict: dict):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config_or_path_or_dir):
        config = None

        if isinstance(config_or_path_or_dir, dict):
            config = config_or_path_or_dir
        else:
            assert isinstance(config_or_path_or_dir, str) or isinstance(config_or_path_or_dir, Path) 

        path_or_dir = Path(config_or_path_or_dir)
        if path_or_dir.is_dir():
            config_path = path_or_dir / 'model_config.json'
            config = utils.Json.load(config_path)
        else:
            assert path_or_dir.is_file()
            config = utils.Json.load(path_or_dir)

        return cls._from_config(config)

    def get_probs(self, *features):
        logits = self(*features)
        return self.sigmoid(logits)

    def get_loss(self, labels, *features):
        logits = self(*features)
        logits = logits.flatten()
        labels = labels.flatten().to(torch.float32)
        loss = self.loss(logits, labels).mean()
        return loss

    def predict(self, path_predict_docs, dir_output, batch_size, use_cache):
        raise NotImplementedError

    @staticmethod
    def _par_decode_doc(predicted_doc, threshold):
        sents = []
        for predicted_sent in predicted_doc['sents']:
            tokens = consts.LM_TOKENIZER.convert_ids_to_tokens(predicted_sent['ids'])
            predicted_spans = [s for s in predicted_sent['spans'] if s[2] > threshold]
            predicted_spans = sorted(predicted_spans, key=lambda s: (s[1] - s[0], s[2]), reverse=True)
            idxs_taken = set()
            spans = []
            for l_idx, r_idx, prob in predicted_spans:
                idxs_set = set(range(l_idx, r_idx + 1))
                if idxs_set & idxs_taken:
                    continue
                idxs_taken |= idxs_set
                phrase = consts.roberta_tokens_to_str(tokens[l_idx: r_idx + 1])
                spans.append([l_idx, r_idx, phrase])
            sents.append({'tokens': tokens, 'spans': spans})
        return sents

    @staticmethod
    def decode(path_predicted_docs, output_dir, threshold, use_cache, use_tqdm):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        path_output = output_dir / (f'doc2sents-{threshold}-' + path_predicted_docs.stem + '.json')
        if use_cache and utils.IO.is_valid_file(path_output):
            print(f'[Decode] Use cache: {path_output}')
            return path_output
        utils.Log.info(f'Decode: {path_output}')
        path_predicted_docs = Path(path_predicted_docs)
        predicted_docs = utils.Pickle.load(path_predicted_docs)
        to_iterate = tqdm(predicted_docs, ncols=100, desc='Decode') if use_tqdm else predicted_docs

        doc2sents = {doc['_id_']: BaseModel._par_decode_doc(doc, threshold) for doc in to_iterate}

        utils.OrJson.dump(doc2sents, path_output)

        # after decode
        decoded_corpus = DecodedCorpus(path_output)
        decoded_corpus.dump_html()
        return path_output

    @staticmethod
    def _par_get_doc_cands(predicted_doc, threshold, filter_punc=True):
        cands = set()
        for predicted_sent in predicted_doc['sents']:
            tokens = consts.LM_TOKENIZER.convert_ids_to_tokens(predicted_sent['ids'])
            predicted_spans = predicted_sent['spans']
            for l_idx, r_idx, prob in predicted_spans:
                if prob > threshold:
                    cand = consts.roberta_tokens_to_str(tokens[l_idx: r_idx + 1])
                    cand = utils.stem_cand(cand)
                    if cand:
                        cands.add(cand)
        if filter_punc:
            cands = {c for c in cands if not (c[0] in PUNCS or c[-1] in PUNCS)}

        return list(cands)

    @ staticmethod
    def get_doc2cands(path_predicted_docs, output_dir, expected_num_cands_per_doc, use_cache, use_tqdm):
        output_dir = Path(output_dir)
        path_output = output_dir / (f'doc2cands-{expected_num_cands_per_doc}-' + path_predicted_docs.stem + '.json')
        if use_cache and utils.IO.is_valid_file(path_output):
            print(f'[Doc2cands] Use cache: {path_output}')
            return path_output
        utils.Log.info(f'Doc2cands: {path_output}')
        path_predicted_docs = Path(path_predicted_docs)
        predicted_docs = utils.Pickle.load(path_predicted_docs)
        to_iterate = tqdm(predicted_docs, ncols=100, desc='Decode') if use_tqdm else predicted_docs
        threshold_l = 0.0
        threshold_r = 1.0
        min_average_num_cands = expected_num_cands_per_doc - .5
        max_average_num_cands = expected_num_cands_per_doc + .5
        for _ in range(20):
            threshold = (threshold_l + threshold_r) / 2
            doc2cands = {doc['_id_']: BaseModel._par_get_doc_cands(doc, threshold) for doc in to_iterate}
            num_cands = [len(cands) for doc, cands in doc2cands.items() if doc in consts.DOCIDS_WITH_GOLD]
            average_num_cands = utils.mean(num_cands)
            print(f'threshold={threshold:.3f} num_cands={average_num_cands}')
            if min_average_num_cands <= average_num_cands <= max_average_num_cands:
                print('threshold OK!')
                break
            if average_num_cands < min_average_num_cands:
                threshold_r = threshold
            else:
                assert max_average_num_cands < average_num_cands
                threshold_l = threshold
        print(f'path_output: {path_output}')
        utils.Json.dump(doc2cands, path_output)
        return path_output

    @ staticmethod
    def load_ckpt(path_ckpt):
        ckpt = torch.load(path_ckpt, map_location='cpu')
        return ckpt['model']


class DecodedCorpus:
    def __init__(self, path_decoded_doc2sents):
        self.path_decoded_doc2sents = Path(path_decoded_doc2sents)
        self.decoded_doc2sents = utils.Json.load(path_decoded_doc2sents)

    def dump_html(self):
        path_output = self.path_decoded_doc2sents.with_name(self.path_decoded_doc2sents.name.replace('doc2sents', 'html')).with_suffix('.html')
        html_lines = []
        for doc, sents in self.decoded_doc2sents.items():
            html_lines.append(f'DOC {doc}')
            for sent in sents:
                # ipdb.set_trace()
                tokens = sent['tokens']
                for l, r, _ in sent['spans']:
                    tokens[l] = consts.HTML_BP + tokens[l]
                    tokens[r] = tokens[r] + consts.HTML_EP + ' |'
                html_lines.append(consts.roberta_tokens_to_str(tokens))
        html_lines = [f'<p>{line}<p>' for line in html_lines]
        utils.TextFile.dumplist(html_lines, path_output)


class BaseFeatureExtractor:
    def __init__(self, output_dir, use_cache=True) -> None:
        super().__init__()
        self.use_cache = use_cache
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    @ staticmethod
    def _get_batch_size(seqlen):
        return 64 * 128 // seqlen

    @ staticmethod
    def _batchify(marked_sents, is_train=False):
        batches = []  # each element is a tuple of (input_ids_batch, input_masks_batch)
        pointer = 0
        total_num = len(marked_sents)
        while pointer < total_num:
            maxlen = len(marked_sents[pointer]['ids'])
            batch_size = BaseFeatureExtractor._get_batch_size(maxlen)
            input_ids_batch = []
            input_masks_batch = []
            pos_spans_batch = []
            neg_spans_batch = []
            possible_spans_batch = []
            for marked_sent in marked_sents[pointer: pointer + batch_size]:
                input_id = marked_sent['ids']
                word_idxs = marked_sent['widxs']
                if not is_train:
                    possible_spans = utils.get_possible_spans(word_idxs, len(input_id), consts.MAX_WORD_GRAM, consts.MAX_SUBWORD_GRAM)
                    possible_spans_batch.append(possible_spans)
                len_diff = maxlen - len(input_id)
                assert len_diff >= 0, 'Input ids must have been sorted!'
                input_ids_batch.append([consts.LM_TOKENIZER.bos_token_id] + input_id + [consts.LM_TOKENIZER.pad_token_id] * len_diff)
                input_masks_batch.append([1] + [1] * len(input_id) + [0] * len_diff)
                if is_train:
                    pos_spans = marked_sent['pos_spans']
                    neg_spans = marked_sent['neg_spans']
                    pos_spans_batch.append(pos_spans)
                    neg_spans_batch.append(neg_spans)
            batch_size = len(input_ids_batch)
            pointer += batch_size
            input_ids_batch = torch.tensor(input_ids_batch)
            input_masks_batch = torch.tensor(input_masks_batch)
            if is_train:
                batches.append((input_ids_batch, input_masks_batch, pos_spans_batch, neg_spans_batch))
            else:
                batches.append((input_ids_batch, input_masks_batch, possible_spans_batch))
        return batches
