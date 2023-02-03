import json
import re

input_raw_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpShoes/shoes.jsonl"
input_annotation_file = "/shared/data2/ppillai3/test/UCPhrase-exp/data/kpShoes/shoes.annotations.jsonl"
output_file = "/shared/data2/ppillai3/test/UCPhrase-exp/src/initial_data_conversion/test_tagging_doc"

# process
with open(input_raw_file) as rf:
    lines = rf.read().splitlines()
data_raw_bullet_points = [json.loads(l) for l in lines]

# process
with open(input_annotation_file) as rf:
    lines = rf.read().splitlines()
data_annotations = [json.loads(l) for l in lines]

bullet_points_dict = {}
for entry in data_raw_bullet_points:
    asin = entry['asin']
    bullet_points_dict[asin] = entry["bullet_point"]

annotations_dict = {}
for entry in data_annotations:
    asin = entry['asin']
    entities = []
    bullet_points = str(bullet_points_dict[asin])[1:-1]
    for entity in entry["entities"]:
        if re.search(entity['value'], bullet_points):
            entity_span = re.search(entity['value'], bullet_points).span()
            entities.append([entity_span[0], entity_span[1], entity['value']])
    annotations_dict[asin] = entities

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(annotations_dict, f, ensure_ascii=False, indent=2)