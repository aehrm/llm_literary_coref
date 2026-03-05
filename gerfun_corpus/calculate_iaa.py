import itertools
import json
import re
from pathlib import Path

import pandas

from llm_literary_coref.eval.evaluator import Evaluator
from llm_literary_coref.mention import Mention, parse_mentions


def load_annotations(path):
    df = pandas.read_csv(path, sep='\t', index_col='i', keep_default_na=False)
    yield from parse_mentions(df['gold'])

def main():
    iaa_dir = Path(__file__).parent / "annotated_tsv" / "iaa"

    evaluator = Evaluator()

    iaa_files = list(iaa_dir.glob("*_*.tsv"))

    keyfn = lambda x: re.search(r'^[^_]+_(.*)\.tsv$', x.name).group(1)
    for doc_name, pair in itertools.groupby(sorted(iaa_files, key=keyfn), keyfn):
        pair = list(sorted(pair))
        assert len(pair) == 2, f"There are more than two documents with the same suffix {doc_name}."

        print(f"Comparing {pair[0].name} with {pair[1].name}")
        mentions_a = list(load_annotations(pair[0]))
        mentions_b = list(load_annotations(pair[1]))
        evaluator.add_document(mentions_a, mentions_b, doc_id=doc_name)

    report = evaluator.report(print_support="all", print_individual_doc_scores=False)
    print(report)

    with open(Path(__file__).parent / "evaluation_reports"/ "iaa_report.json", "w") as f:
        json.dump(evaluator.report(as_dict=True), f, indent=2)
        print(f"written to {f.name}")




if __name__ == "__main__":
    main()
