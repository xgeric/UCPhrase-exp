import utils
import consts
import string
import functools
import pickle
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from preprocess.preprocess import Preprocessor
from preprocess.annotator_base import BaseAnnotator


MINCOUNT = 2
MINGRAMS = 2
MAXGRAMS = consts.MAX_WORD_GRAM


PUNCS_SET = set(string.punctuation) - {'-'}
STPWD_SET = set(utils.TextFile.readlines('../data/stopwords.txt'))
with open('../data/products_found_in_staging.pkl', 'rb') as f:
    CUSTOM_KEYPHRASES = pickle.load(f)

print("LENTH OF CUSTOM KEYPHRASES", len(CUSTOM_KEYPHRASES))
print("CUSTOM_KEYPHRASES", CUSTOM_KEYPHRASES)


@functools.lru_cache(maxsize=100000)
def is_valid_ngram(ngram: list):
    valid_ngram = True
    for token in ngram:
        # Discard digit tokens, empty tokens and STOPWORDS
        if not token or token in STPWD_SET or token.isdigit():
            valid_ngram = False
    # Set of characters of the ngram
    charset = set(''.join(ngram))
    # Empty set of characters or intersection with punctuactions
    if not charset or (charset & (PUNCS_SET)):
        valid_ngram = False
    # ngrams such as 'networks-' are not valid
    if ngram[0].startswith('-') or ngram[-1].endswith('-'):
        valid_ngram = False

    if ' '.join(ngram) in CUSTOM_KEYPHRASES and not valid_ngram:
        print(f"ngram {ngram} is not valid")
    return valid_ngram


class CoreAnnotator(BaseAnnotator):
    def __init__(self, preprocessor: Preprocessor, use_cache):
        super().__init__(preprocessor, use_cache=use_cache)

    @staticmethod
    def _par_mine_doc_phrases(doc_tuple):
        tokenized_doc, tokenized_id_doc = doc_tuple
        assert tokenized_doc['_id_'] == tokenized_id_doc['_id_']
        assert len(tokenized_doc['sents']) == len(tokenized_id_doc['sents'])

        # Create keyphrases for each document
        phrase2cnt = Counter()
        phrase2instances = defaultdict(list)
        for i_sent, (sent, sent_dict) in enumerate(zip(tokenized_doc['sents'], tokenized_id_doc['sents'])):
            tokens = sent.lower().split()
            widxs = sent_dict['widxs']
            num_words = len(widxs)
            widxs.append(len(tokens))  # for convenience
            # Iterave over all ngrams of length MINGRAMS to MAXGRAMS
            # First iteration: [an ontology]
            # Second iteration: [an ontology modelling]
            # ...
            # Store frequency of the ngrams
            for n in range(MINGRAMS, MAXGRAMS + 2):
                for i_word in range(num_words - n + 1):
                    l_idx = widxs[i_word]
                    r_idx = widxs[i_word + n] - 1
                    ngram = tuple(tokens[l_idx: r_idx + 1])
                    # Ġme chat ronics Ġapproach -> 'mechatronics', 'approach'
                    ngram = tuple(''.join(ngram).split(consts.GPT_TOKEN.lower())[1:])

                    # If ngram is User Keyphrase, add it to the phrase2cnt and phrase2instances
                    if ' '.join(ngram) in CUSTOM_KEYPHRASES:
                        phrase = ' '.join(ngram)
                        phrase2cnt[phrase] += 1
                        phrase2instances[phrase].append([i_sent, l_idx, r_idx])
                        continue

                    if is_valid_ngram(ngram):
                        phrase = ' '.join(ngram)
                        phrase2cnt[phrase] += 1
                        # Store the index of the sentence, the start and end
                        phrase2instances[phrase].append([i_sent, l_idx, r_idx])

        # Filter phrases with less than MINCOUNT instances
        phrases = []
        for phrase, count in phrase2cnt.items():
            # If phrase is user keyphrase, don't filter it
            if phrase in CUSTOM_KEYPHRASES:
                phrases.append(phrase)
                continue

            if count >= MINCOUNT:
                phrases.append(phrase)

        phrases = sorted(phrases, key=lambda p: len(p), reverse=True)
        cleaned_phrases = set()
        for p in phrases:
            has_longer_pattern = False
            for cp in cleaned_phrases:
                if p in cp:
                    has_longer_pattern = True
                    break
            if not has_longer_pattern and len(p.split()) <= MAXGRAMS:
                cleaned_phrases.add(p)
        phrase2instances = {p: phrase2instances[p] for p in cleaned_phrases}
        return phrase2instances

    def _mark_corpus(self):
        # Load tokenizer dataset
        tokenized_docs = utils.JsonLine.load(self.path_tokenized_corpus)
        tokenized_id_docs = utils.JsonLine.load(self.path_tokenized_id_corpus)
        # Parallelize mining of phrases
        phrase2instances_list = utils.Process.par(
            func=CoreAnnotator._par_mine_doc_phrases,
            iterables=list(zip(tokenized_docs, tokenized_id_docs)),
            num_processes=consts.NUM_CORES,
            desc='[CoreAnno] Mine phrases'
        )
        doc2phrases = dict()
        for i_doc, doc in tqdm(list(enumerate(tokenized_id_docs)), ncols=100, desc='[CoreAnno] Tag docs'):
            for s in doc['sents']:
                s['phrases'] = []
            phrase2instances = phrase2instances_list[i_doc]
            doc2phrases[doc['_id_']] = list(phrase2instances.keys())
            for phrase, instances in phrase2instances.items():
                for i_sent, l_idx, r_idx in instances:
                    doc['sents'][i_sent]['phrases'].append([[l_idx, r_idx], phrase])
        utils.Json.dump(doc2phrases, self.dir_output / f'doc2phrases.{self.path_tokenized_corpus.stem}.json')

        return tokenized_id_docs
