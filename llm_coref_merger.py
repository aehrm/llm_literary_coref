import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator, Dict, Type, List

import pandas as pd

from omegaconf import OmegaConf

from llm_literary_coref.llm_annotator import LLMRunner
from llm_literary_coref.mention import Mention, parse_mentions
from llm_literary_coref.prompts.merge_prompts import MergePrompt, BasicMergePrompt
from llm_literary_coref.util import JSONEncoder

PROMPT_REGISTRY: Dict[str, Type[MergePrompt]] = {
    "default": BasicMergePrompt,
    # TODO droc
}


def load_document(input_files: List[Path]) -> pd.DataFrame:
    """Load and concatenate TSV files into a single DataFrame."""
    dfs = []
    for f in input_files:
        print(f"loading {f}")
        df = pd.read_csv(f, sep="\t", index_col='i')
        df['is_section_start'] = 0
        df.loc[df.index[0], 'is_section_start'] = 1
        dfs.append(df)

    all_df = pd.concat(dfs)
    return all_df.sort_index()


def merge_section(input_files: List[Path], annotator: LLMRunner, prompt_class: Type[MergePrompt], output_file: Path):
    document_df = load_document(input_files)

    if not {'token', 'pred'} <= set(document_df.columns):
        raise ValueError("Input file must contain columns 'token' and 'pred'.")

    tokens = document_df['token']
    mentions: List[Mention] = list(parse_mentions(document_df['pred']))

    prompt = prompt_class(tokens, document_df['is_section_start'], mentions)
    res = annotator.run(prompt)

    # output res as debug output
    if res:
        debug_filename = output_file.parent / f"{output_file.stem}_debug.json"
        with open(debug_filename, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, cls=JSONEncoder)
        print(f"Saved debug output to {debug_filename}")

    if res.exception:
        print(f"Annotator failed: {res.exception}")
        sys.exit(1)


    decoded_mentions = res.output
    output_df = document_df.copy().drop('is_section_start', axis='columns')
    output_df['pred'] = ''

    for mention in decoded_mentions:
        for k in mention.token_idx:
            output_df.loc[k, 'pred'] = output_df.loc[k, 'pred'] + str(mention)

    output_df.to_csv(output_file, sep='\t')
    print(f"Saved annotations to {str(output_file)}")






def main():
    parser = argparse.ArgumentParser(description="Run LLM-based annotation on TSV text sections.")
    parser.add_argument('--input_files', type=Path, nargs='+', required=True, help='List of input TSV files to process.' )
    parser.add_argument('--output_file', type=Path, required=True, help='Directory to save output TSV and debug JSON files.' )
    parser.add_argument('--model', type=str, required=True, help='LLM Model string (e.g., "openai/gpt-4-turbo", "anthropic/claude-3-opus").')
    parser.add_argument('--prompt_type', type=str, default="default", help=f'Key for the prompt class to use. Options: {list(PROMPT_REGISTRY.keys())}' )
    parser.add_argument('-X', '--generation_args', type=str, required=False, action='append', help='Generation arguments.')
    args = parser.parse_args()

    # args.output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.generation_args and len(args.generation_args) > 0:
        gen_args = OmegaConf.from_dotlist(args.generation_args)
    else:
        gen_args = OmegaConf.create(dict(
            seed=123,
            temperature=0,
            reasoning=dict(enabled=False)
        ))

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

    merge_section(args.input_files, annotator, prompt_class, args.output_file)


if __name__ == "__main__":
    main()