import json
import os.path
import random
import tempfile
from typing import Optional

import click
import re
import spacy
from spacy_langdetect import LanguageDetector
from google.cloud import bigquery
from nltk import sent_tokenize


nlp_model = spacy.load("en_core_web_md")
nlp_model.add_pipe(LanguageDetector(), name="language_detector", last=True)

# TODO: Filter based on review sources - create dictionary
with open('./data/review_sources_patterns.txt') as f:
    review_sources_patterns = f.read().splitlines()

print(review_sources_patterns)


regexes = []
for pattern in review_sources_patterns:
    regexes.append(r"\b{p}\b".format(p=pattern))
# Creates a regex that matches if any of our regexes match.
combined = "(" + ")|(".join(regexes) + ")"


def detect_lang(text: str) -> str:
    try:
        doc = nlp_model(text)
        lang = doc._.language["language"]
    except Exception as e:
        print(f"Exception in lang detection, text: {text} - {e}")
        lang = "UNK"
    return lang


def clean_text(text: str) -> str:
    """Function to remove html tags and multiple line breaks from the content"""
    clean_text = re.sub(re.compile("<.*?>"), " ", text)
    clean_text = re.sub("\\s+", " ", clean_text)
    return clean_text.replace("\n", " ").replace("\r", " ")


@click.command()
@click.option("--size", required=False, default=100000, type=int)
@click.option("--output_dir", required=False, type=str)
@click.option("--test_size", required=False, default=0.1, type=float)
@click.option("--bigquery_table", required=False, default="staging_klue_reviews", type=str)
def prepare_dataset(size: int, output_dir: Optional[str], test_size: float, bigquery_table: str):
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="klue_")

    client = bigquery.Client()

    query = f"""
        SELECT *
        FROM `klue-1265.datawarehouse.{bigquery_table}`
        WHERE RAND() < {size}/(SELECT COUNT(*) FROM `klue-1265.datawarehouse.{bigquery_table}`)
    """
    print(query)

    query_job = client.query(query)

    standard_folder = os.path.join(output_dir, "standard")
    os.makedirs(standard_folder, exist_ok=True)

    with open(
        os.path.join(standard_folder, "klue.tagging.docs.jsonl"), "w"
    ) as tagging_docs_file, open(
        os.path.join(standard_folder, "klue.train.jsonl"), "w"
    ) as train_file, open(
        os.path.join(standard_folder, "klue.test.jsonl"), "w"
    ) as test_file:
        for row in query_job:
            # Avoid reviews in a different language than English
            detected_lang = detect_lang(row["content"])
            if detected_lang != "en":
                print(f"Skipped review {row['id']} - Detected language {detected_lang}")
                continue

            full_content = ""
            if row["title"]:
                full_content += clean_text(row["title"]) + ".\n"
            if row["content"]:
                full_content += clean_text(row["content"])

            # Filter patterns
            matches = re.findall(combined, full_content)
            for match in matches:
                # Find matches
                idx = list(map(bool, match)).index(True)
                match = review_sources_patterns[idx]
                full_content = full_content.replace(match, ".")

            row_dict = {"_id_": row["id"], "sents": sent_tokenize(full_content)}

            tagging_docs_file.write(json.dumps(row_dict) + "\n")
            if random.random() < test_size:
                test_file.write(json.dumps(row_dict) + "\n")
            else:
                train_file.write(json.dumps(row_dict) + "\n")

    with open(os.path.join(standard_folder, "stem.doc2references.json"), "w") as f:
        json.dump({}, f)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(
            {
                "lm_name": "allenai/cs_roberta_base",
                "path_tagging_docs": "./standard/klue.tagging.docs.jsonl",
                "paths_tagging_human": [],
                "path_test": "./standard/klue.test.jsonl",
                "path_train": "./standard/klue.train.jsonl",
                "path_phrase": "../wiki_quality.txt",
                "path_stopwords": "../stopwords.txt",
                # The ones below aren't needed, but their code requires them
                "path_stem_test": "./standard/stem.klue.test.jsonl",
                "path_stem_train": "./standard/stem.klue.train.jsonl",
                "path_stem_doc2references": "./standard/stem.doc2references.json",
                "kp_num_candidates_per_doc": 18.3,
            },
            f,
        )

    print(
        f"Dataset be saved at: {output_dir}. Copy it inside the data folder of UCPhrase-exp"
    )


if __name__ == "__main__":
    prepare_dataset()
