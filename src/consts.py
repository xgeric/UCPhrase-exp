import torch
import utils
import argparse
import transformers
from pathlib import Path
from functools import lru_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--dir_data", type=str, default='../data/devdata/')
    parser.add_argument("--exp_prefix", type=str, default='', help='additional exp name')
    parser.add_argument("--model_prefix", type=str, default='', help='additional model name')
    parser.add_argument("--path_model_config", type=str, default='../configs/core.CNN.3layers.json')
    args = parser.parse_args()
    return args


class DataConfig:
    def __init__(self, path_config_json):
        path_config_json = Path(path_config_json)
        config_dict = utils.Json.load(path_config_json)
        dir_config = path_config_json.parent

        self.lm_name = config_dict['lm_name']
        self.path_test = dir_config / config_dict['path_test']
        self.path_train = dir_config / config_dict['path_train']
        self.path_phrase = dir_config / config_dict['path_phrase']
        self.path_tagging_docs = dir_config / config_dict['path_tagging_docs']
        self.paths_tagging_human = [dir_config / p for p in config_dict['paths_tagging_human']]
        self.path_stem_test = dir_config / config_dict['path_stem_test']
        self.path_stem_train = dir_config / config_dict['path_stem_train']
        self.path_stem_doc2references = dir_config / config_dict['path_stem_doc2references']
        self.dir_config = dir_config
        self.kp_num_candidates_per_doc = config_dict['kp_num_candidates_per_doc']

    def todict(self):
        return {key: str(val) for key, val in self.__dict__.items()}


ARGS = parse_args()
utils.Log.info(f'ARGS: {ARGS.__dict__}')

# data
DIR_DATA = Path(ARGS.dir_data)
PATH_DATA_CONFIG = DIR_DATA / 'config.json'
DATA_CONFIG = DataConfig(PATH_DATA_CONFIG)
PATH_MODEL_CONFIG = Path(ARGS.path_model_config)
STEM_DOC2REFS = {doc: {g for g in golds if len(g.split()) > 1} for doc, golds in utils.Json.load(DATA_CONFIG.path_stem_doc2references).items()}
DOCIDS_WITH_GOLD = {doc for doc, golds in STEM_DOC2REFS.items() if golds}  # Task 2 evaluation is performed on docs with gold keyphrases

# huggingface LM
GPT_TOKEN = 'Ġ'
LM_NAME = DATA_CONFIG.lm_name
LM_NAME_SUFFIX = LM_NAME.split('/')[-1]
DEVICE = utils.get_device(ARGS.gpu)
LM_MODEL = transformers.RobertaModel.from_pretrained(LM_NAME).eval().to(DEVICE)
LM_TOKENIZER = transformers.RobertaTokenizerFast.from_pretrained(LM_NAME)
print(f'[consts] Loading pretrained model: {LM_NAME} OK!')

# html visualization
HTML_BP = '<span style=\"color: #ff0000\">'
HTML_EP = '</span>'

# settings
MAX_SENT_LEN = 64
MAX_WORD_GRAM = 5
MAX_SUBWORD_GRAM = 10
# Number of negative candidates
# NEGATIVE_RATIO = 1 - positive_keyphrases = 3 | then: negative_keyphrases = 3
NEGATIVE_RATIO = 1

# multiprocessing
NUM_CORES = 1
torch.set_num_threads(NUM_CORES)


def roberta_tokens_to_str(tokens):
    return ''.join(tokens).replace(GPT_TOKEN, ' ').strip()
