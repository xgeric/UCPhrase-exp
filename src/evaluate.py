import ipdb
import utils
import consts
from functools import lru_cache
from tqdm import tqdm
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer

import string
PUNCS = set(string.punctuation) - {'-'}
STEMMER = SnowballStemmer('porter', ignore_stopwords=False)


@lru_cache(maxsize=100000)
def STEM_WORD(w):
    return STEMMER.stem(w)


class Evaluator:
    def __init__(self, filter_punc=True, use_stem=False):
        self.use_stem = use_stem
        self.filter_punc = filter_punc

    def evaluate(self, path_doc2cands):
        utils.Log.info(f'[Eval] {path_doc2cands}')
        doc2golds = consts.STEM_DOC2REFS

        path_doc2cands = Path(path_doc2cands)
        doc2cands = utils.Json.load(path_doc2cands)
        assert set(doc2cands.keys()) - consts.DOCIDS_WITH_GOLD == set(doc2golds.keys()) - consts.DOCIDS_WITH_GOLD
        num_cands = 0
        info_list = []
        macro_recalls = []
        micro_num_gold = 0
        micro_num_hits = 0
        for doc in tqdm(sorted(doc2golds.keys()), ncols=100, desc='eval'):
            golds: set = doc2golds[doc]
            if not golds:
                assert doc not in consts.DOCIDS_WITH_GOLD, ipdb.set_trace()
                continue

            cands = doc2cands[doc]
            if self.use_stem:
                cands = {utils.stem_cand(c) for c in cands}

            if self.filter_punc:
                cands = {c for c in cands if not (c[0] in PUNCS or c[-1] in PUNCS)}
            hits = cands & golds
            missed = golds - hits
            recall = len(hits) / len(golds)
            if missed:
                info_list.append(f'{doc} {missed} {len(cands)} ')
            macro_recalls.append(recall)
            micro_num_gold += len(golds)
            micro_num_hits += len(hits)
            num_cands += len(cands)
        average_num_cands = num_cands / len(macro_recalls)
        average_micro_recall = micro_num_hits / micro_num_gold
        average_macro_recall = sum(macro_recalls) / len(macro_recalls)

        path_doc2cands = Path(path_doc2cands)
        path_output = path_doc2cands.with_name('eval.' + path_doc2cands.name)
        eval_result = {
            'metrics': {
                'num_cands': average_num_cands,
                'macro_recall': average_macro_recall,
                'micro_recall': average_micro_recall,
            },
            'infos': info_list,
        }
        utils.Log.info(f'num cands: {average_num_cands}')
        utils.Log.info(f'macro recall: {average_macro_recall}')
        utils.Log.info(f'micro recall: {average_micro_recall}')
        utils.Json.dump(eval_result, path_output)
        return path_output


class SentEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def _to_char_offsets(sent):
        tokens = []
        spans = []
        for splitsent in sent:
            _tokens = splitsent['tokens']
            if len(_tokens) > 0:
                if not _tokens[0].startswith(consts.GPT_TOKEN):
                    _tokens[0] = consts.GPT_TOKEN + _tokens[0]
            _spans = splitsent['spans']
            _spans = [(_span[0] + len(tokens), _span[1] + len(tokens) + 1, _span[2]) for _span in _spans]
            tokens.extend(_tokens)
            spans.extend(_spans)
        span_start_to_offset = {}
        span_end_to_offset = {}
        n_tokens = []
        assert len(tokens) == 0 or tokens[0].startswith(consts.GPT_TOKEN)
        n_chars = 0
        span_start_to_offset[0] = 0
        for i, token in enumerate(tokens):
            if token.startswith(consts.GPT_TOKEN):
                if len(n_tokens) != 0:
                    n_chars += len(n_tokens[-1])
                    span_end_to_offset[i] = n_chars
                    n_chars += 1
                    span_start_to_offset[i] = n_chars
                n_tokens.append(token.replace(consts.GPT_TOKEN, ''))
            else:
                n_tokens[-1] = n_tokens[-1] + token
        if len(n_tokens) != 0:
            span_end_to_offset[len(tokens)] = n_chars + len(n_tokens[-1])
        text = " ".join(n_tokens)
        spans = list(map(lambda x: (span_start_to_offset[x[0]], span_end_to_offset[x[1]], x[2]), spans))
        # assert all(x[2] == text[x[0]: x[1]] for x in spans)
        return spans

    def evaluate(self, path_decoded_doc2sents, paths_doc2golds, *args):
        utils.Log.info(f'[Eval] {path_decoded_doc2sents}')
        doc2sents = utils.Json.load(path_decoded_doc2sents)
        num_predict = 0
        num_gold = 0
        num_match = 0
        all_spans = []
        pred_extra_spans = []
        gold_extra_spans = []
        for path in paths_doc2golds:
            doc2golds = utils.Json.load(path)
            for doc in tqdm(sorted(doc2golds.keys()), ncols=100, desc='eval'):
                gold = doc2golds[doc]
                sent = doc2sents[doc]
                spans = self._to_char_offsets(sent)
                gold_spans = set([(l, r, w) for l, r, w in gold])
                pred_spans = set([(l, r, w) for l, r, w in spans])
                all_spans.append([sorted(list(gold_spans)), sorted(list(pred_spans))])
                num_match += len(gold_spans & pred_spans)
                num_predict += len(pred_spans)
                num_gold += len(gold_spans)
                pred_extra_spans.extend(list(map(lambda x: x[2], list(pred_spans - gold_spans))))
                gold_extra_spans.extend(list(map(lambda x: x[2], list(gold_spans - pred_spans))))
        path_decoded_doc2sents = Path(path_decoded_doc2sents)
        path_output = path_decoded_doc2sents.with_name(path_decoded_doc2sents.name.replace('doc2sents', 'eval_local'))
        precision = 1. * num_match / num_predict
        recall = 1. * num_match / num_gold
        f1 = 2 * precision * recall / (precision + recall)
        eval_result = {
            'metrics': {
                'num_predict': num_predict,
                'num_gold': num_gold,
                'num_match': num_match,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            "pred_extra_spans": pred_extra_spans,
            "gold_extra_spans": gold_extra_spans
        }
        utils.Log.info(f'precision: {precision}')
        utils.Log.info(f'recall: {recall}')
        utils.Log.info(f'f1: {f1}')
        utils.Json.dump(eval_result, path_output)
        return path_output
