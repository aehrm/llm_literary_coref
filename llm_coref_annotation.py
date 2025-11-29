import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, Type

import pandas as pd

from omegaconf import OmegaConf

from llm_literary_coref.llm_annotator import LLMRunner
from llm_literary_coref.mention import Mention
from llm_literary_coref.prompts.mention_prompts import MentionPromptBasic, MentionPrompt
from llm_literary_coref.util import JSONEncoder

PROMPT_REGISTRY: Dict[str, Type[MentionPrompt]] = {
    "default": MentionPromptBasic,
    # TODO droc
}



def get_mention_spans(mentions: pd.Series) -> Iterator[Mention]:
    id_: str
    for id_, g in mentions.groupby(mentions):
        if id_ == '' or pd.isna(id_):
            continue

        yield Mention(str(id_), token_idx=list(g.index), references=[])



def annotate_section(section_path: Path, annotator: LLMRunner, prompt_class: Type[MentionPrompt], output_dir: Path):
    section_df = pd.read_csv(section_path, sep='\t', keep_default_na=False, index_col='i')

    if not {'token', 'mention'} <= set(section_df.columns):
        raise ValueError("Input file must contain columns 'token' and 'mention'.")

    tokens = section_df['token']
    mention_spans = list(get_mention_spans(section_df['mention']))

    print(f'Starting annotations for {section_path.name}: {len(tokens)} tokens, {len(mention_spans)} mentions found.')

    if not mention_spans:
        res = None
    else:
        prompt = prompt_class(tokens, mention_spans)
        res = annotator.run(prompt)

    # output res as debug output
    if res:
        debug_filename = output_dir / f"{section_path.stem}_debug.json"
        with open(debug_filename, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, cls=JSONEncoder)
        print(f"Saved debug output to {debug_filename}")

    if res.exception:
        print(f"Annotator failed for {section_path.name}: {res.exception}")
        sys.exit(1)


    decoded_mentions = res.output
    output_df = section_df.copy().drop('mention', axis='columns')
    output_df['pred'] = ''

    for mention in decoded_mentions:
        for k in mention.token_idx:
            output_df.loc[k, 'pred'] = output_df.loc[k, 'pred'] + str(mention)

    output_filename = output_dir / f"{section_path.stem}_processed.tsv"
    output_df.to_csv(output_filename, sep='\t')
    print(f"Saved annotations to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM-based annotation on TSV text sections.")
    parser.add_argument('--input_files', type=Path, nargs='+', required=True, help='List of input TSV files to process.' )
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory to save output TSV and debug JSON files.' )
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

    for input_path in args.input_files:
        annotate_section(input_path, annotator, prompt_class, args.output_dir)


if __name__ == "__main__":
    main()