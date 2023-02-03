import json
from Levenshtein import distance
from difflib import SequenceMatcher

input_annotation_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpCoffee/coffee.annotations.jsonl"
input_results_file = "/shared/data2/ppillai3/test/UCPhrase-exp/experiments/kpCoffee-cs_roberta_base-core.CNN.3layers/model/tagging.decoded.epoch-40/doc2sents-0.95-tokenized.id.kpCoffee.tagging.docs.json"
input_oamine_results_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpCoffee/standard/kpCoffee.candidateAsins.jsonl"

output_file = "/shared/data2/ppillai3/test/UCPhrase-exp/src/initial_data_conversion/metrics_output"

# process
with open(input_annotation_file) as rf:
    lines = rf.read().splitlines()
data_annotations = [json.loads(l) for l in lines]

with open(input_results_file) as rf:
    data_results = json.load(rf)
#     lines = rf.read().splitlines()
# data_results = [json.loads(l) for l in lines]

with open(input_oamine_results_file) as rf:
    lines = rf.read().splitlines()
data_oamine_results = [json.loads(l) for l in lines]

annotations_dict = {}
for entry in data_annotations:
    asin = entry['asin']
    entities = []
    for entity in entry["entities"]:
        entities.append(entity['value'].lower())
    annotations_dict[asin] = entities

results_dict = {}
for asin in data_results:
    entities = []
    for sent in data_results[asin]:
        for span in sent['spans']:
            # print(span[2])
            entities.append(span[2].lower())
    results_dict[asin] = entities

oamine_results_dict = {}
for entry in data_oamine_results:
    asin = entry['_id_']
    oamine_results_dict[asin] = [candidate.lower() for candidate in entry['candidates']]
# print(oamine_results_dict)

# calculate document-level recall
output_corpus_recall = {}

# def longest_common_substring(s1: str, s2: str) -> str:
#     """Computes the longest common substring of s1 and s2"""
#     seq_matcher = SequenceMatcher(isjunk=None, a=s1, b=s2)
#     match = seq_matcher.find_longest_match(0, len(s1), 0, len(s2))
#     if match.size:
#         return s1[match.a : match.a + match.size]
#     else:
#         return ""

for asin in annotations_dict:
    if asin not in results_dict:
        continue
    annotation = annotations_dict[asin]
    results = results_dict[asin]
    oamine_results = oamine_results_dict[asin]
    # print(asin)
    # print("Annotation:", annotation)
    # print("Experimental result:", results)
    results_str = " ".join(results).lower()

    matched = 0
    total = 0

    for phrase in annotation:
        # yet_phrase_matched = False
        # for word in phrase.lower().split():
        #     # print(word)
        #     if word in results_str:
        #         yet_phrase_matched = True
        #         break
        # if yet_phrase_matched:
        #     matched += 1
        comparison_score = 0
        for oamine_result in oamine_results:
            s1 = phrase
            s2 = oamine_result
            if min(len(s1), len(s2)) <= 0:
                continue
            # levenshtein similarity
            comparison_score = max(comparison_score, 1. - distance(s1, s2) / min(len(s1), len(s2)))
            # longest common substring
            # comparison_score = len(longest_common_substring(s1, s2)) / min(len(s1), len(s2))
        for result in results:
            s1 = phrase
            s2 = result
            if min(len(s1), len(s2)) <= 0:
                continue
            # levenshtein similarity
            comparison_score = max(comparison_score, 1. - distance(s1, s2) / min(len(s1), len(s2)))        
            # longest common substring
            # comparison_score = len(longest_common_substring(s1, s2)) / min(len(s1), len(s2))
        if comparison_score > 0.4:
            matched += 1
        total += 1
        # print(matched)
        # print(total)
    output_corpus_recall[asin] = [matched, total]
    


# calculate corpus level recall
corpus_matched = 0
corpus_total = 0
for asin in output_corpus_recall:
    corpus_matched += output_corpus_recall[asin][0]
    corpus_total += output_corpus_recall[asin][1]
corpus_recall = corpus_matched / corpus_total


# calculate corpus level precision
corpus_matched_precision = 0
corpus_total_precision = 0
for asin in results_dict:
    if asin not in annotations_dict:
        continue
    annotation = annotations_dict[asin]
    results = results_dict[asin]
    oamine_results = oamine_results_dict[asin]
    annotations_str = " ".join(annotation).lower()

    # matched = 0
    # total = 0
    for phrase in results:
        comparison_score = 0
        for ann in annotation:
            s1 = phrase
            s2 = ann
            if min(len(s1), len(s2)) <= 0:
                continue
            # levenshtein similarity
            comparison_score = max(comparison_score, 1. - distance(s1, s2) / min(len(s1), len(s2)))        
            if comparison_score > 0.4:
                corpus_matched_precision += 1
            corpus_total_precision += 1
corpus_precision = corpus_matched_precision / corpus_total_precision

# write results
with open(output_file, 'w') as jsonl_output:
    jsonl_output.write("Corpus-level recall: " + str(corpus_recall) + '\n')
    jsonl_output.write("Corpus-level precision: " + str(corpus_precision) + '\n')
    for asin in output_corpus_recall:
        matched = output_corpus_recall[asin][0]
        total = output_corpus_recall[asin][1]
        # json.dump(str(asin) + ": " + str(matched / total), jsonl_output)
        jsonl_output.write(str(matched/total))
        jsonl_output.write('\n')