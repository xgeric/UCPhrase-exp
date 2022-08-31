import pickle
import json

with open('/Users/alex/Documents/Klue/Projects/UCPhrase-exp/data/products_found_in_staging.pkl', 'rb') as f:
    CUSTOM_KEYPHRASES = pickle.load(f)

with open(
        '/Users/alex/Documents/Klue/Projects/UCPhrase-exp/data/exp_custom_keyphrases/standard/preprocess-cs_roberta_base_V2/annotate.CoreAnnotator/doc2phrases.tokenized.klue.train.json') as json_file:
    json_with_keyphrases = json.load(json_file)

with open(
        '/Users/alex/Documents/Klue/Projects/UCPhrase-exp/data/exp_custom_keyphrases/standard/preprocess-cs_roberta_base/annotate.CoreAnnotator/doc2phrases.tokenized.klue.train.json') as json_file:
    json_without_keyphrases = json.load(json_file)

for el in json_without_keyphrases:
    with_ = json_with_keyphrases[el]
    without_ = json_without_keyphrases[el]
    for w in without_:
        if w not in with_:
            print(el)
            print("w", with_)
            print("without", without_)
            print("-")
