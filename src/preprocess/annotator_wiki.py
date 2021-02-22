import utils
import consts
from tqdm import tqdm
from pathlib import Path
from match import KeywordProcessor
from preprocess.annotator_base import BaseAnnotator

MARK_PREFIX = '<23fwa<'
MARK_SUFFIX = '>23fwa>'


class WikiAnnotator(BaseAnnotator):
    def __init__(self, preprocessor, use_cache, path_standard_phrase):
        super().__init__(preprocessor, use_cache=use_cache)
        self.path_standard_phrase = Path(path_standard_phrase)
        self.path_tokenized_phrase = self.dir_output / f'tokenized.{self.path_standard_phrase.name}'

    def tokenize_phrases(self):
        if self.use_cache and utils.IO.is_valid_file(self.path_tokenized_phrase):
            print(f'[WikiAnno] Use cache: {self.path_tokenized_phrase}')
            return self.path_tokenized_phrase
        phrases = utils.TextFile.readlines(self.path_standard_phrase)
        phrases = [' ' + p for p in phrases if len(p.split()) > 1]  # for Roberta tokenizer
        tqdm_phrases = tqdm(phrases, ncols=100, desc=f'[WikiAnno] {self.path_standard_phrase}')
        tokenized_phrases = [' '.join(consts.LM_TOKENIZER.tokenize(phrase, add_special_tokens=False)) for phrase in tqdm_phrases]
        utils.TextFile.dumplist(tokenized_phrases, self.path_tokenized_phrase)
        return self.path_tokenized_phrase

    def _mark_corpus(self):
        self.tokenize_phrases()
        kwprocessor = KeywordProcessor()
        quality_phrases = utils.TextFile.readlines(self.path_tokenized_phrase)
        keyword_dict = {f'{MARK_PREFIX}{p}{MARK_SUFFIX}': [p] for p in quality_phrases if p.startswith(consts.GPT_TOKEN)}
        kwprocessor.add_keywords_from_dict(keyword_dict)

        # find out quality phrases in the corpus
        raw_corpus = utils.TextFile.load(self.path_tokenized_corpus)
        replaced_corpus = kwprocessor.replace_keywords(raw_corpus)
        replaced_corpus = replaced_corpus.replace(f'{consts.GPT_TOKEN}{MARK_PREFIX}', f'{MARK_PREFIX}{consts.GPT_TOKEN}')
        path_replaced_corpus = self.path_marked_corpus.with_name(f'tmp.replaced.{self.path_marked_corpus.name}')
        utils.TextFile.dump(replaced_corpus, path_replaced_corpus)
        del kwprocessor

        # generate marked html, for debugging only
        path_replaced_html = path_replaced_corpus.with_suffix('.html')
        replaced_html = replaced_corpus.replace(MARK_PREFIX, consts.HTML_BP).replace(MARK_SUFFIX, consts.HTML_EP).replace(consts.GPT_TOKEN, '')
        replaced_html = '\n'.join([f'<p> {line} </p>' for line in replaced_html.splitlines()])
        utils.TextFile.dump(replaced_html, path_replaced_html)
        del replaced_html

        # mark the positions of quality phrases
        num_finally_matched_phrases = 0
        num_partially_matched_phrases = 0
        raw_id_docs = utils.JsonLine.load(self.path_tokenized_id_corpus)
        raw_docs = [utils.Json.loads(line) for line in raw_corpus.splitlines()]
        replaced_docs = [utils.Json.loads(line) for line in replaced_corpus.splitlines()]
        assert len(raw_docs) == len(replaced_docs) == len(raw_id_docs)
        for raw_doc, raw_id_doc, replaced_doc in tqdm(zip(raw_docs, raw_id_docs, replaced_docs), ncols=100, total=len(raw_docs), desc='Clean phrases'):
            raw_sents = raw_doc['sents']
            raw_id_sents = raw_id_doc['sents']
            replaced_sents = replaced_doc['sents']
            assert len(raw_sents) == len(replaced_sents) == len(raw_id_sents)
            for raw_sent, raw_id_sent, replaced_sent in zip(raw_sents, raw_id_sents, replaced_sents):
                raw_tokens = raw_sent.split()
                replaced_tokens = replaced_sent.split()
                assert len(raw_tokens) == len(replaced_tokens)
                tmp_phrase = [[-1, -1], '']
                tmp_in_phrase = False
                phrases = []
                for token_i, replaced_token in enumerate(replaced_tokens):
                    raw_token = raw_tokens[token_i]
                    if replaced_token.startswith(MARK_PREFIX):
                        tmp_in_phrase = True
                        tmp_phrase[0][0] = token_i
                        tmp_phrase[1] = raw_token
                    elif replaced_token.endswith(MARK_SUFFIX):
                        assert tmp_in_phrase
                        tmp_in_phrase = False
                        tmp_phrase[0][1] = token_i
                        tmp_phrase[1] += ' ' + raw_token
                        phrases.append(tmp_phrase)
                        tmp_phrase = [[-1, -1], '']
                    elif tmp_in_phrase:
                        tmp_phrase[1] += ' ' + raw_token
                # clean partially matched phrases
                widxs_set = set(raw_id_sent['widxs'])
                num_original_phrases = len(phrases)
                cleaned_phrases = [p for p in phrases if p[0][0] in widxs_set and (p[0][1] + 1 in widxs_set or p[0][1] == len(raw_tokens) - 1)]
                raw_id_sent['phrases'] = cleaned_phrases
                num_cleaned_phrases = len(cleaned_phrases)
                num_partially_matched_phrases += num_original_phrases - num_cleaned_phrases
                num_finally_matched_phrases += num_cleaned_phrases
        print(f'Finally matched phrases: {num_finally_matched_phrases}')
        print(f'Partially matched phrases: {num_partially_matched_phrases}')

        marked_docs = raw_id_docs
        return marked_docs
