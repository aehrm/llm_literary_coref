import argparse
from pathlib import Path

import pandas

from llm_literary_coref.eval.evaluator import Evaluator
from llm_literary_coref.mention import parse_mentions


def main():


    parser = argparse.ArgumentParser(description="Evaluate LLM inference results.")
    parser.add_argument('--pred_file', type=Path, required=True, help='Path to the prediction TSV file. Must contain columns "pred" and "gold".')
    args = parser.parse_args()


    evaluator = Evaluator(filter_condition="any")

    merged_doc_df = pandas.read_csv(args.pred_file, sep='\t', index_col='i')
    sys_mentions = list(parse_mentions(merged_doc_df['pred']))
    key_mentions = list(parse_mentions(merged_doc_df['gold']))
    evaluator.add_document(key_mentions=key_mentions, sys_mentions=sys_mentions)

    report = evaluator.report(print_support="key", print_individual_doc_scores=False)
    print(report)

    pass

if __name__ == "__main__":
    main()