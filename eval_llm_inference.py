import argparse
import json
from pathlib import Path

import pandas

from llm_literary_coref.eval.evaluator import Evaluator
from llm_literary_coref.mention import parse_mentions

def evaluate_files(pred_files, json_report=None):
    document_level_evaluator = Evaluator()
    section_level_evaluator = Evaluator()

    for pred_file in pred_files:
        print(f"Evaluating {pred_file.name}")
        merged_doc_df = pandas.read_csv(pred_file, sep='\t', index_col='i', keep_default_na=False)
        if not 'pred' in merged_doc_df.columns or not 'gold' in merged_doc_df.columns:
            raise ValueError(f"Prediction file {pred_file.name} does not contain columns 'pred' and 'gold'.")

        sys_mentions = list(parse_mentions(merged_doc_df['pred']))
        key_mentions = list(parse_mentions(merged_doc_df['gold']))

        document_level_evaluator.add_document(key_mentions=key_mentions, sys_mentions=sys_mentions, doc_id=pred_file.stem)

        if 'is_section_start' in merged_doc_df.columns:
            section_num = merged_doc_df['is_section_start'].cumsum()
            for section_id, section in merged_doc_df.groupby(section_num):
                print(f"Evaluating {pred_file.name}, section {section_id}")
                sys_mentions = list(parse_mentions(section['pred']))
                key_mentions = list(parse_mentions(section['gold']))
                section_level_evaluator.add_document(key_mentions=key_mentions, sys_mentions=sys_mentions, doc_id=f"{pred_file.stem}_{section_id:03d}")

    print('#######')
    print('Document-level evaluation:')
    print('#######')
    print(document_level_evaluator.report(print_support="key", print_individual_doc_scores=len(document_level_evaluator.seen_documents) > 1))

    if len(section_level_evaluator.seen_documents) > 0:
        print('\n\n\n#######')
        print('Section-level evaluation:')
        print('#######')
        print(section_level_evaluator.report(print_support="key", print_individual_doc_scores=False))

    if json_report:
        with open(json_report, 'w') as f:
            document_report = document_level_evaluator.report(as_dict=True)
            section_report = section_level_evaluator.report(as_dict=True)
            output_obj = {
                'document_level': document_report,
                'section_level': section_report,
            }
            json.dump(output_obj, f, indent=2)
            print(f"written to {f.name}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM inference results.")
    parser.add_argument('--pred_files', type=Path, nargs='+', required=True, help='Path to the prediction TSV file. Must contain columns "pred" and "gold".')
    parser.add_argument('--write_json_report', type=Path, required=False, help='Path to write JSON report to.')
    args = parser.parse_args()
    evaluate_files(args.pred_files, json_report=args.write_json_report)

if __name__ == "__main__":
    main()