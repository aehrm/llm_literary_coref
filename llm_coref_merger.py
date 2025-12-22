import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Iterator, Dict, Type, List

import pandas as pd

from omegaconf import OmegaConf

from llm_literary_coref.llm_annotator import LLMRunner
from llm_literary_coref.mention import Mention, parse_mentions
from llm_literary_coref.openrouter import OPENROUTER_PROVIDER
from llm_literary_coref.prompts.merge_prompts import MergePrompt, BasicMergePrompt
from llm_literary_coref.util import JSONEncoder

PROMPT_REGISTRY: Dict[str, Type[MergePrompt]] = {
    "default": BasicMergePrompt,
}


def load_document(gold_file: Path, input_files: List[Path]) -> pd.DataFrame:
    print(f"loading gold file {gold_file}")
    gold_df = pd.read_csv(gold_file, sep="\t", index_col='i', keep_default_na=False)

    all_inference_dfs = []
    for f in input_files:
        print(f"loading {f}")
        df = pd.read_csv(f, sep="\t", index_col='i', keep_default_na=False)
        #df['is_section_start'] = 0
        #df.loc[df.index[0], 'is_section_start'] = 1
        all_inference_dfs.append(df)

    inference_df = pd.concat(all_inference_dfs).sort_index()
    gold_df['pred'] = inference_df['pred']
    if not all(~gold_df['pred'].isna()):
        missing_indices = list(sorted(gold_df.index[gold_df['pred'].isna()]))
        ranges = [(k, k + len(list(g)) - 1) for k, g in itertools.groupby(enumerate(missing_indices), lambda x: x[1] - x[0])]
        print("WARN: for the following tokens, no prediction has been supplied: " + ', '.join([f"{a}-{b}" for a, b in ranges]))
    
    return gold_df.sort_index()


def merge_section(gold_file: Path, input_files: List[Path], annotator: LLMRunner, prompt_class: Type[MergePrompt], output_file: Path):
    document_df = load_document(gold_file, input_files)

    if not {'token', 'pred'} <= set(document_df.columns):
        raise ValueError("Input file must contain columns 'token' and 'pred'.")

    tokens = document_df['token']
    mentions: List[Mention] = list(parse_mentions(document_df['pred']))

    prompt = prompt_class(tokens, document_df['is_section_start'], mentions)
    res = annotator.run(prompt)

    # output res as debug output
    if res:
        debug_filename = output_file.parent / f"{output_file.stem}.json"
        with open(debug_filename, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, cls=JSONEncoder)
        print(f"Saved debug output to {debug_filename}")

    if res.exception:
        print(f"Annotator failed: {res.exception}")
        sys.exit(1)


    decoded_mentions = res.output
    output_df = document_df.copy()
    output_df['pred'] = ''

    for mention in decoded_mentions:
        for k in mention.token_idx:
            output_df.loc[k, 'pred'] = output_df.loc[k, 'pred'] + str(mention)

    output_df.to_csv(output_file, sep='\t')
    print(f"Saved annotations to {str(output_file)}")






def main():
    parser = argparse.ArgumentParser(description="Run LLM-based annotation on TSV text sections. Groups input files according to their name")
    parser.add_argument('--gold_files', type=Path, nargs='+', required=True, help='List of gold TSV files to process.' )
    parser.add_argument('--input_files', type=Path, nargs='+', required=True, help='List of input TSV files to process.' )
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory for merged TSV files.' )
    parser.add_argument('--model', type=str, required=True, help='LLM Model string (e.g., "openai/gpt-4-turbo", "anthropic/claude-3-opus").')
    parser.add_argument('--prompt_type', type=str, default="default", help=f'Key for the prompt class to use. Options: {list(PROMPT_REGISTRY.keys())}' )
    parser.add_argument('-X', '--generation_args', type=str, required=False, action='append', help='Generation arguments.')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.generation_args and len(args.generation_args) > 0:
        gen_args = OmegaConf.from_dotlist(args.generation_args)
    else:
        gen_args = OmegaConf.create(dict(
            seed=123,
            temperature=0,
            reasoning=dict(enabled=False)
        ))

        model_provider = OPENROUTER_PROVIDER.get(args.model)
        if model_provider:
            gen_args.provider = dict(only=[model_provider])
        else:
            print(f'warn: no default provider is known for model {args.model}; results may not be deterministic!')

    print('Picked up the following generation arguments:')
    print(OmegaConf.to_yaml(gen_args))

    prompt_class = PROMPT_REGISTRY.get(args.prompt_type, None)
    if prompt_class is None:
        print(f"Invalid prompt type: {args.prompt_type}. Options: {list(PROMPT_REGISTRY.keys())}")
        sys.exit(1)

    annotator = LLMRunner(
        model=args.model,
        request_args=OmegaConf.to_container(gen_args, resolve=True)
    )

    tsv_files: List[Path] = [f for f in args.input_files if f.name.endswith('.tsv')]
    keyfn = lambda x: re.sub(r'_section_[0-9]+\.tsv', '', x.name)
    for document_name, section_files in itertools.groupby(sorted(tsv_files, key=keyfn), key=keyfn):
        section_files = list(section_files)
        print(args.gold_files)
        gold_file = [f for f in args.gold_files if f.name == f"{document_name}.tsv"]
        if len(gold_file) != 1:
            print(f'for inference TSV files {section_files}, no gold file named "{document_name}.tsv" specified')
            sys.exit(1)

        gold_file = gold_file[0]
        output_file = args.output_dir / f"{document_name}.tsv"
        merge_section(gold_file, list(section_files), annotator, prompt_class, output_file)


if __name__ == "__main__":
    main()
