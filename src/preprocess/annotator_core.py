from ossaudiodev import SOUND_MIXER_TREBLE
import utils
import consts
import string
import functools
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from preprocess.preprocess import Preprocessor
from preprocess.annotator_base import BaseAnnotator
from flashtext import KeywordProcessor ##added this one


MINCOUNT = 2
MINGRAMS = 2
MAXGRAMS = consts.MAX_WORD_GRAM


PUNCS_SET = set(string.punctuation) - {'-'}
STPWD_SET = set(utils.TextFile.readlines('../data/stopwords.txt'))


@functools.lru_cache(maxsize=100000)
def is_valid_ngram(ngram: list):
    for token in ngram:
        if not token or token in STPWD_SET or token.isdigit():
            return False
    charset = set(''.join(ngram))
    if not charset or (charset & (PUNCS_SET)):
        return False
    if ngram[0].startswith('-') or ngram[-1].endswith('-'):
        return False
    return True


class CoreAnnotator(BaseAnnotator):
    def __init__(self, preprocessor: Preprocessor, use_cache):
        super().__init__(preprocessor, use_cache=use_cache)

    @staticmethod
    def _par_mine_doc_phrases(doc_tuple):
        tokenized_doc, tokenized_id_doc = doc_tuple
        assert tokenized_doc['_id_'] == tokenized_id_doc['_id_']
        assert len(tokenized_doc['sents']) == len(tokenized_id_doc['sents'])

        phrase2cnt = Counter()
        phrase2instances = defaultdict(list)
        for i_sent, (sent, sent_dict) in enumerate(zip(tokenized_doc['sents'], tokenized_id_doc['sents'])):
            tokens = sent.lower().split()
            widxs = sent_dict['widxs']
            num_words = len(widxs)
            widxs.append(len(tokens))  # for convenience
            for n in range(MINGRAMS, MAXGRAMS + 2):
                for i_word in range(num_words - n + 1):
                    l_idx = widxs[i_word]
                    r_idx = widxs[i_word + n] - 1
                    ngram = tuple(tokens[l_idx: r_idx + 1])
                    ngram = tuple(''.join(ngram).split(consts.GPT_TOKEN.lower())[1:])
                    if is_valid_ngram(ngram):
                        phrase = ' '.join(ngram)
                        phrase2cnt[phrase] += 1
                        phrase2instances[phrase].append([i_sent, l_idx, r_idx])
        phrases = [phrase for phrase, count in phrase2cnt.items() if count >= MINCOUNT]
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
    
    ## our method to replace theirs
    @staticmethod
    def _par_mine_doc_phrases_NEW(doc_tuple):
        #not sure if necessary - take in input information and make sure IDs and number of docs match up
        tokenized_doc, tokenized_id_doc, Ecandidate_doc, segmented_sentences_doc = doc_tuple

        assert tokenized_doc['_id_'] == tokenized_id_doc['_id_']        
        assert len(tokenized_doc['sents']) == len(tokenized_id_doc['sents'])

        # if (tokenized_doc['_id_'] != candidate_doc['_id_']):
        #     return defaultdict(list)

        phrase2cnt = Counter()
        phrase2instances = defaultdict(list)

        candidates_docs = utils.JsonLine.load("/shared/data2/ppillai3/test/UCPhrase-exp/data/kpWater/standard/kpWater.candidateAsins.jsonl")
        candidate_doc = ""
        candidates = []
        for doc in candidates_docs:
            if (doc['_id_'] == tokenized_doc['_id_']):
                # print(doc['_id_'])
                candidates = doc['candidates']
                # print(candidates)
                break
        if (len(candidates) == 0):
            return defaultdict(list)
        # return phrase2instances


        
        # for candidate in candidate_doc:
        #     print(candidate['candidates'])
        # return phrase2instances

        ##THIS IS OAMINE-EXTENSION ADAPTATION OF UCPHRASE
        #bug fix candidate doc
        # phrases_initial = candidate_doc
        # phrases_next = {'vals': phrases_initial}
        # candidate_doc = phrases_next['vals']

        # keywords_found = []
        # #use phrases from candidate and match to the raw sentence doc
        # keyword_processor = KeywordProcessor()
        # keyword_processor.add_keywords_from_list(candidate_doc)

        # phrase2cnt = Counter()
        # phrase2instances = defaultdict(list)
        # for i_sent, (sent, sent_dict, candidates, sent_segmented) in enumerate(zip(tokenized_doc['sents'], tokenized_id_doc['sents'], candidate_doc, segmented_sentences_doc['sents'])):
        #     tokens = sent.lower().split()
        #     widxs = sent_dict['widxs']
        #     num_words = len(widxs)
        #     widxs.append(len(tokens))  # for convenience
        #     keywords_found.extend(keyword_processor.extract_keywords(sent_segmented.lower()))
        #     # print(keywords_found)
        #     for n in range(MINGRAMS, MAXGRAMS + 2):
        #         for i_word in range(num_words - n + 1):
        #             l_idx = widxs[i_word]
        #             r_idx = widxs[i_word + n] - 1
        #             ngram = tuple(tokens[l_idx: r_idx + 1])
        #             ngram = tuple(''.join(ngram).split(consts.GPT_TOKEN.lower())[1:])
        #             # if (ngram == )
        #             # print(tokens[l_idx: r_idx + 1])
        #             # if is_valid_ngram(ngram):
        #             for keyword in keywords_found:
        #                 if keyword in ngram:
        #                     phrase = ' '.join(ngram)
        #                     # print(phrase)
        #                     phrase2cnt[phrase] += 1
        #                     phrase2instances[phrase].append([i_sent, l_idx, r_idx])
        # phrases = [phrase for phrase, count in phrase2cnt.items() if count >= MINCOUNT]
        # phrases = sorted(phrases, key=lambda p: len(p), reverse=True)
        # cleaned_phrases = set()
        # for p in phrases:
        #     has_longer_pattern = False
        #     for cp in cleaned_phrases:
        #         if p in cp:
        #             has_longer_pattern = True
        #             break
        #     if not has_longer_pattern and len(p.split()) <= MAXGRAMS:
        #         cleaned_phrases.add(p)
        #     phrase2instances = {p: phrase2instances[p] for p in cleaned_phrases}

        #     return phrase2instances
        # return phrase2instances



        # keyword_processor = KeywordProcessor()
        # keyword_processor.add_keywords_from_list(candidate_doc['candidates'])

        # keywords_found = []
        # for s in segmented_sentences_doc['sents']:
        #     #not exactly sure on how extend works, so if keywords overlap this might be destroying that info
        #     keywords_found.extend(keyword_processor.extract_keywords(s.lower()))

        # for keyword in keywords_found:
        #     print(keyword)

        ##THIS IS JUST UCPHRASE
        phrase2cnt = Counter()
        phrase2instances = defaultdict(list)
        for i_sent, (sent, sent_dict) in enumerate(zip(tokenized_doc['sents'], tokenized_id_doc['sents'])):
            tokens = sent.lower().split()
            widxs = sent_dict['widxs']
            num_words = len(widxs)
            widxs.append(len(tokens))  # for convenience
            for n in range(MINGRAMS, MAXGRAMS + 2):
                for i_word in range(num_words - n + 1):
                    l_idx = widxs[i_word]
                    r_idx = widxs[i_word + n] - 1
                    ngram = tuple(tokens[l_idx: r_idx + 1])
                    ngram = tuple(''.join(ngram).split(consts.GPT_TOKEN.lower())[1:])
                    if is_valid_ngram(ngram):
                        phrase = ' '.join(ngram)
                        # print(" ".join(ngram) + " is valid phrase")
                        for candidate in candidates:
                            # print("candidate is " + candidate)
                            # print(phrase + " is valid phrase")
                            if (candidate in phrase):
                                # print(ngram)
                                phrase2cnt[phrase] += 1
                                phrase2instances[phrase].append([i_sent, l_idx, r_idx])
                                candidates.remove(candidate)
                                break
        phrases = [phrase for phrase, count in phrase2cnt.items() if count >= MINCOUNT]
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
        return phrase2instances


        ##THIS IS TO SEE DOC2PHRASES
        #set up phrase2instances
        phrase2cnt = Counter()
        phrase2instances = defaultdict(list)

        if (len(tokenized_doc['sents']) == 0):
            return phrase2instances
        
        #use phrases from candidate and match to the raw sentence doc
        keyword_processor = KeywordProcessor()
        keyword_processor.add_keywords_from_list(candidate_doc)

        keywords_found = []
        for s in segmented_sentences_doc['sents']:
            #not exactly sure on how extend works, so if keywords overlap this might be destroying that info
            keywords_found.extend(keyword_processor.extract_keywords(s.lower()))
        
        # for i_sent, (sent, sent_dict, candidates, sent_segmented) in enumerate(zip(tokenized_doc['sents'], tokenized_id_doc['sents'], candidate_doc['sents'], segmented_sentences_doc['sents'])):
        #     widxs = sent_dict['widxs']
        #     num_words = len(widxs)
        #     for n in range(MINGRAMS, MAXGRAMS + 2):
        #         for i_word in range(num_words - n + 1):
        #             # word = sent_segmented[i_word]
        #             l_idx = widxs[i_word]
        #             r_idx = widxs[i_word + n] - 1
        #             # if ()
        
        
        for keywords in keywords_found:
            phrase = str(keywords)
            #made up numbers for the ids below, FIX THIS LATER
            phrase2instances[phrase].append([0, 0, 1])


        return phrase2instances
        
        # for doc in segmented_sentences_docs:
        #     if doc['_id_'] == tokenized_doc['_id_']:
        #         segmented_sentences_doc = doc
        #         # print(tokenized_doc['_id_'])
        #         # print(raw_sentences_doc['_id_'])
        #         keywords_found = []
        #         for s in segmented_sentences_doc['sents']:
        #             #not exactly sure on how extend works, so if keywords overlap this might be destroying that info
        #             keywords_found.extend(keyword_processor.extract_keywords(s.lower()))
                
        #         # print(keywords_found)

        #         for keywords in keywords_found:
        #             phrase = str(keywords)
        #             #made up numbers for the ids below, FIX THIS LATER
        #             phrase2instances[phrase].append([0, 0, 1])

        #         return phrase2instances
        #         #correct doc has been found, so we can exit
        #         break
        
        # return phrase2instances

    def _mark_corpus(self):
        tokenized_docs = utils.JsonLine.load(self.path_tokenized_corpus)
        tokenized_id_docs = utils.JsonLine.load(self.path_tokenized_id_corpus)
        segmented_sentences_docs = utils.JsonLine.load("/shared/data2/ppillai3/test/UCPhrase-exp/data/kpWater/standard/kpWater.train.jsonl")

        ## for later, load in the raw segmented sentences and then feed into par NEW
        # raw_sentences_docs = utils.JsonLine.load("/shared/data2/ppillai3/test/UCPhrase-exp/data/kpWater/standard/kpWater.train.jsonl")

        #for later, load in the  sentences and then feed into par NEW
        candidates_docs = utils.JsonLine.load("/shared/data2/ppillai3/test/UCPhrase-exp/data/kpWater/standard/kpWater.candidateAsins.jsonl")

        # phrase2instances_list = utils.Process.par(
        #     func=CoreAnnotator._par_mine_doc_phrases,
        #     iterables=list(zip(tokenized_docs, tokenized_id_docs)),
        #     num_processes=consts.NUM_CORES,
        #     desc='[CoreAnno] Mine phrases'
        # )

        # # figuring out what's going on
        # print("Phrase2instances_list is " + str(len(phrase2instances_list)))
        # count = 0
        # for item in phrase2instances_list:
        #     if count > 50:
        #         break
        #     print("\n New item")
        #     print("\n _________________ \n Phrase2instances_list item is " + str(item))
        #     count += 1

        ## We will change phrase2instances_list instead of this entire _mark_corpus method
        phrase2instances_list = utils.Process.par(
            func=CoreAnnotator._par_mine_doc_phrases_NEW,
            iterables=list(zip(tokenized_docs, tokenized_id_docs, candidates_docs, segmented_sentences_docs)),
            num_processes=consts.NUM_CORES,
            desc='[CoreAnno] Mine phrases'
        )

        doc2phrases = dict()
        for i_doc, doc in tqdm(list(enumerate(tokenized_id_docs)), ncols=100, desc='[CoreAnno] Tag docs'):
            for s in doc['sents']:
                s['phrases'] = []
            try:
                phrase2instances = phrase2instances_list[i_doc]
            except:
                #bad practice, but patchwork will break out of loop if this doesn't work
                # print("Length of phrase2instances_list is " + str(len(phrase2instances_list)))
                # print("i_doc is " + str(i_doc))
                break
            doc2phrases[doc['_id_']] = list(phrase2instances.keys())
            for phrase, instances in phrase2instances.items():
                for i_sent, l_idx, r_idx in instances:
                    doc['sents'][i_sent]['phrases'].append([[l_idx, r_idx], phrase])
        utils.Json.dump(doc2phrases, self.dir_output / f'doc2phrases.{self.path_tokenized_corpus.stem}.json')

        return tokenized_id_docs

    ## Was our method to replace their _mark_corpus
    ## isn't fully necessary, we can just replace the _par_mine_doc_phrases_NEW method
    # def _mark_corpus_NEW(self):
    #     tokenized_docs = utils.JsonLine.load(self.path_tokenized_corpus)
    #     tokenized_id_docs = utils.JsonLine.load(self.path_tokenized_id_corpus)
        
    #     doc2phrases = dict()
    #     for i_doc, doc in tqdm(list(enumerate(tokenized_id_docs)), ncols=100, desc='[CoreAnno] Tag docs'):
    #         for s in doc['sents']:
    #             s['phrases'] = []
        
    #     return tokenized_id_docs
