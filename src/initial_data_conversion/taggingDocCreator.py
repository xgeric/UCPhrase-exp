import json
from pathlib import Path

# input_test_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpBreakfast_Cereal/standard/breakfast_cereal.train.jsonl"
# output_tag_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpBreakfast_Cereal/standard/breakfast_cereal.tagging.docs.jsonl"


# input_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpBreakfast_Cereal/standard/breakfast_cereal.candidateAsins.jsonl"
# def load_jsonl(fname):
#   f_obj = Path(fname)
#   docs = []
#   with f_obj.open("r") as f:
#     for line in f:
#       docs.append(json.loads(line))
#   return docs
# candidate_asin_data = load_jsonl(input_file)

# asin_set = set()
# for doc in candidate_asin_data:
#     val = doc["_id_"]
#     asin_set.add(val)

# with open(input_test_file) as rf:
#     lines = rf.read().splitlines()
# data = [json.loads(l) for l in lines]

# with open(output_tag_file, 'w') as jsonl_output:
#     for entry in data:
#         if len(entry['sents']) == 0 or entry['_id_'] not in asin_set:
#             continue
#         json.dump(entry, jsonl_output)
#         jsonl_output.write('\n')

# output_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpBreakfast_Cereal/standard/breakfast_cereal.tagging.human_0.json"

input_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpLaundry_Detergent/standard/preprocess-cs_roberta_base/annotate.CoreAnnotator/tokenized.laundry_detergent.train.jsonl"
output_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpLaundry_Detergent/standard/laundry_detergent.tagging.human_0.json"

# data = JsonLine.load(input_file)
with open(input_file) as rf:
    lines = rf.read().splitlines()
data = [json.loads(l) for l in lines]


output_data = {}
for i in range(len(data)):
    asin = data[i]['_id_']
    phrases = []
    
    sents = data[i]['sents']
    for sent in sents:
        try:
            cleaned_phrases_entry = []
            phrases_entry = sent['phrases']
            for phrase in phrases_entry:
                # if (asin == "B0814BBV3W"):
                cleaned_phrase = [phrase[0][0], phrase[0][1], phrase[1]]
                phrases.append(cleaned_phrase)            
            # phrases.append(cleaned_phrases_entry)
        except:
            pass
    # if (asin == "B0814BBV3W"):
    #     print(phrases)

    output_data[asin] = (phrases)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)