import utils
import consts
import random
from tqdm import tqdm
from pathlib import Path
from preprocess.preprocess import Preprocessor


class BaseAnnotator:
    def __init__(self, preprocessor: Preprocessor, use_cache=True):
        self.use_cache = use_cache
        self.preprocessor = preprocessor
        self.dir_output = self.preprocessor.dir_preprocess / f'annotate.{self.__class__.__name__}'
        self.dir_output.mkdir(exist_ok=True)
        self.path_tokenized_corpus = self.preprocessor.path_tokenized_corpus
        self.path_tokenized_id_corpus = self.preprocessor.path_tokenized_id_corpus
        self.path_marked_corpus = self.dir_output / f'{self.path_tokenized_corpus.name}'

    @staticmethod
    def _par_sample_train_data(marked_doc):
        print("Processing sents with POS and NEG samples - base")
        sents = marked_doc['sents']
        # Here, we should insert the negative keyphrases such as `product solving`
        for sent in sents:
            phrases = sent['phrases']
            assert phrases
            positive_spans = [tuple(phrase[0]) for phrase in phrases]
            num_positive = len(positive_spans)
            # sample negatives
            word_idxs = sent['widxs']
            # Iterave over all ngrams of length MINGRAMS to MAXGRAMS
            # (0, 4), (1, 5)...
            # (0, 3), (1, 4), ...
            # (0, 2), (1, 3), ...
            # (0, 1), (1, 2), ...
            all_spans = utils.get_possible_spans(word_idxs, len(sent['ids']), consts.MAX_WORD_GRAM, consts.MAX_SUBWORD_GRAM)
            # Discard positive spans in negative_spans
            possible_negative_spans = set(all_spans) - set(positive_spans)
            num_negative = min(len(possible_negative_spans), int(num_positive * consts.NEGATIVE_RATIO))
            sampled_negative_spans = random.sample(possible_negative_spans, k=num_negative)
            sent['pos_spans'] = positive_spans
            sent['neg_spans'] = sampled_negative_spans
            sent.pop('phrases')
        return marked_doc

    def sample_train_data(self):
        print("SAMPLE TRAIN DATA - base")
        assert utils.IO.is_valid_file(self.path_marked_corpus)

        path_output = self.dir_output / f'sampled.neg{consts.NEGATIVE_RATIO}.{self.path_marked_corpus.name}'
        if self.use_cache and utils.IO.is_valid_file(path_output):
            print(f'[SampleTrain] Use cache: {path_output}')
            return path_output

        marked_docs = utils.JsonLine.load(self.path_marked_corpus)
        sampled_docs = [BaseAnnotator._par_sample_train_data(d) for d in tqdm(marked_docs, ncols=100, desc='[Sample Train]')]
        utils.JsonLine.dump(sampled_docs, path_output)
        return path_output

    @staticmethod
    def get_path_sampled_train_spans(path_sampled_train_data):
        print("GET PATH SAMPLED TRAIN - base")
        path_sampled_train_data = Path(path_sampled_train_data)
        path_output = path_sampled_train_data.with_name(f'{path_sampled_train_data.stem}.spans.json')
        if path_output.exists():
            return path_output
        sampled_docs = utils.JsonLine.load(path_sampled_train_data)
        marked_sents = [sent for doc in sampled_docs for sent in doc['sents']]
        marked_sents = sorted(marked_sents, key=lambda s: len(s['ids']), reverse=True)
        spans = []
        for marked_sent in tqdm(marked_sents, ncols=100, desc='[Annotator] get positive spans'):
            word_idxs = marked_sent['widxs']
            swidx2widx = {swidx: widx for widx, swidx in enumerate(word_idxs)}
            swidx2widx.update({len(marked_sent['ids']): len(swidx2widx)})
            for l_idx, r_idx in marked_sent['pos_spans']:
                wl_idx, wr_idx = swidx2widx[l_idx], swidx2widx[r_idx + 1] - 1
                spanlen = wr_idx - wl_idx + 1
                if spanlen > consts.MAX_WORD_GRAM:
                    continue
                assert spanlen > 0
                spans.append((1, marked_sent['ids'][l_idx: r_idx + 1]))
            for l_idx, r_idx in marked_sent['neg_spans']:
                spans.append((0, marked_sent['ids'][l_idx: r_idx + 1]))
        spans = [(label, ''.join(consts.PRETRAINED_TOKENIZER.convert_ids_to_tokens(ids)).replace(consts.GPT_TOKEN, ' ').strip()) for label, ids in spans]
        utils.Json.dump(spans, path_output)
        return path_output

    def _mark_corpus(self):
        raise NotImplementedError

    def mark_corpus(self):
        print("MARK CORPUS - base")
        if self.use_cache and utils.IO.is_valid_file(self.path_marked_corpus):
            print(f'[Annotate] Use cache: {self.path_marked_corpus}')
            return
        # This _mark_corpus is coming from annotator_core
        print("Calling _mark_corpus - base")
        marked_corpus = self._mark_corpus()
        # Remove empty sents and docs - keep sentences with at least one phrase
        print("Remove empty sents - base")
        for raw_id_doc in marked_corpus:
            for sent in raw_id_doc['sents']:
                sent['phrases'] = [p for p in sent['phrases'] if p[0][1] - p[0][0] + 1 <= consts.MAX_SUBWORD_GRAM]
            raw_id_doc['sents'] = [s for s in raw_id_doc['sents'] if s['phrases']]
        marked_corpus = [d for d in marked_corpus if d['sents']]
        print("Store _mark_corpus in : ", self.path_marked_corpus, " - base")
        utils.JsonLine.dump(marked_corpus, self.path_marked_corpus)
