import utils
import consts
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


class Preprocessor:

    def __init__(
            self,
            path_corpus,
            num_cores=8,
            use_cache=False):
        self.use_cache = use_cache
        self.num_cores = num_cores

        # establish preprocess folder
        self.path_corpus = Path(path_corpus)
        self.dir_corpus = self.path_corpus.parent
        self.dir_preprocess = self.dir_corpus / f'preprocess-{consts.LM_NAME_SUFFIX}'
        self.dir_preprocess.mkdir(exist_ok=True)

        # path_tokenized_corpus: wordpieces tokenized with huggingface LM tokenizer
        # path_tokenized_id_corpus: tokenized wordpiece ids with word boundaries
        self.path_tokenized_corpus = self.dir_preprocess / f'tokenized.{self.path_corpus.name}'
        self.path_tokenized_id_corpus = self.dir_preprocess / f'tokenized.id.{self.path_corpus.name}'

    @staticmethod
    def _par_tokenize_doc(doc):
        docid = doc['_id_']
        sents = doc['sents']

        # tokenize
        # NOTE: add space before each raw sentence to tokenize the first token with GPT_TOKEN for phrase matching
        tokenized_sents = [consts.LM_TOKENIZER.tokenize(' ' + s, add_special_tokens=False) for s in sents]
        cleaned_tokenized_sents = []
        for tokens in tokenized_sents:
            tokens_batch = utils.get_batches(tokens, batch_size=consts.MAX_SENT_LEN)
            cleaned_tokenized_sents += tokens_batch
        # print(type(cleaned_tokenized_sents))
        # [print(tokens) for tokens in cleaned_tokenized_sents]
        tokenized_doc = {'_id_': docid, 'sents': [' '.join(tokens) for tokens in cleaned_tokenized_sents]}

        tokenized_id_doc = {'_id_': doc['_id_'], 'sents': []}
        for tokens in cleaned_tokenized_sents:
            widxs = [i for i, token in enumerate(tokens) if token.startswith(consts.GPT_TOKEN)]  # the indices of start of words
            ids = consts.LM_TOKENIZER.convert_tokens_to_ids(tokens)
            tokenized_id_doc['sents'].append({'ids': ids, 'widxs': widxs})

        return tokenized_doc, tokenized_id_doc

    def tokenize_corpus(self):
        # if self.use_cache and utils.IO.is_valid_file(self.path_tokenized_corpus) and utils.IO.is_valid_file(self.path_tokenized_id_corpus):
        #     print(f'[Preprocessor] Use cache: {self.path_tokenized_corpus}')
        #     return
        docs = utils.JsonLine.load(self.path_corpus)
        pool = Pool(processes=self.num_cores)
        pool_func = pool.imap(func=Preprocessor._par_tokenize_doc, iterable=docs)
        doc_tuples = list(tqdm(pool_func, total=len(docs), ncols=100, desc=f'[Tokenize] {self.path_corpus}'))
        tokenized_docs = [doc for doc, iddoc in doc_tuples]
        tokenized_id_docs = [iddoc for doc, iddoc in doc_tuples]
        pool.close()
        pool.join()
        utils.JsonLine.dump(tokenized_docs, self.path_tokenized_corpus)
        utils.JsonLine.dump(tokenized_id_docs, self.path_tokenized_id_corpus)


    ## Our par method for the candidates file
    @staticmethod
    def _par_tokenize_candidates_doc(doc):
        # docid = doc['_id_']
        # sents = doc['sents']
        phrases_initial = doc
        phrases_next = {'vals': phrases_initial}
        phrases = phrases_next['vals']


        # tokenize
        # NOTE: add space before each raw sentence to tokenize the first token with GPT_TOKEN for phrase matching
        tokenized_sents = [consts.LM_TOKENIZER.tokenize(' ' + s, add_special_tokens=False) for s in phrases]
        cleaned_tokenized_sents = []
        for tokens in tokenized_sents:
            tokens_batch = utils.get_batches(tokens, batch_size=consts.MAX_SENT_LEN)
            cleaned_tokenized_sents += tokens_batch
        # tokenized_doc = {'tokens': [' '.join(tokens) for tokens in cleaned_tokenized_sents]}
        tokenized_doc = [' '.join(tokens) for tokens in cleaned_tokenized_sents]

        

        # tokenized_id_doc = {'_id_': doc['_id_'], 'sents': []}
        tokenized_id_doc = {}
        # for tokens in cleaned_tokenized_sents:
        #     widxs = [i for i, token in enumerate(tokens) if token.startswith(consts.GPT_TOKEN)]  # the indices of start of words
        #     ids = consts.LM_TOKENIZER.convert_tokens_to_ids(tokens)
        #     tokenized_id_doc['sents'].append({'ids': ids, 'widxs': widxs})

        return tokenized_doc, tokenized_id_doc
        # return tokenized_doc
    

    ## our method for the candidates file
    def tokenize_candidate_doc(self):

        candidate_docs = utils.JsonLine.load("/shared/data2/ppillai3/test/UCPhrase-exp/data/kpWater/water.chunk.jsonl")
        pool = Pool(processes=self.num_cores)
        pool_func = pool.imap(func=Preprocessor._par_tokenize_candidates_doc, iterable=candidate_docs)
        doc_tuples = list(tqdm(pool_func, total=len(candidate_docs), ncols=100, desc=f'[Tokenize] {self.path_corpus}'))
        tokenized_docs = [doc for doc, iddoc in doc_tuples]
        # # tokenized_id_docs = [iddoc for doc, iddoc in doc_tuples]
        pool.close()
        pool.join()
        utils.JsonLine.dump(tokenized_docs, "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpWater/standard/preprocess-cs_roberta_base/tokenized.water.chunk.jsonl")
        # # utils.JsonLine.dump(tokenized_id_docs, self.path_tokenized_id_corpus)