import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, Type

import pandas as pd

from llm_literay_coref.llm_annotator import LLMMentionAnnotator

from omegaconf import OmegaConf

from llm_literay_coref.mention import Mention
from llm_literay_coref.prompts.mention_prompts import MentionPromptBasic
from llm_literay_coref.prompts.prompt import MentionPrompt
from llm_literay_coref.util import JSONEncoder

PROMPT_REGISTRY: Dict[str, Type[MentionPrompt]] = {
    "default": MentionPromptBasic,
    # TODO droc
}



def get_mention_spans(mentions: pd.Series) -> Iterator[Mention]:
    id_: int|str
    for id_, g in mentions.groupby(mentions):
        if id_ == '' or pd.isna(id_):
            continue

        yield Mention(id_, token_idx=list(g.index), references=[])



def annotate_section(section_path: Path, annotator: LLMMentionAnnotator, prompt: MentionPrompt, output_dir: Path):
    section_df = pd.read_csv(section_path, sep='\t', keep_default_na=False, index_col='i')

    if not {'token', 'mention'} <= set(section_df.columns):
        raise ValueError("Input file must contain columns 'token' and 'mention'.")

    tokens = section_df['token']
    mention_spans = list(get_mention_spans(section_df['mention']))

    if not mention_spans:
        print(f"No mentions found in {section_path.name}, skipping LLM call.")
        res = None
    else:
        res = annotator.run(tokens, mention_spans)

    # output res as debug output
    if res:
        debug_filename = output_dir / f"{section_path.stem}_debug.json"
        with open(debug_filename, 'w', encoding='utf-8') as f:
            json.dump(res, f, default=str, indent=2, cls=JSONEncoder)
        print(f"Saved debug output to {debug_filename}")

    if res.exception:
        print(f"Annotator failed for {section_path.name}: {res.exception}")
        sys.exit(1)


    decoded_mentions = prompt.decode(res.output)
    output_df = section_df.copy()
    output_df['pred'] = ''

    for mention in decoded_mentions:
        for k in mention.token_idx:
            output_df.loc[k, 'pred'] = output_df.loc[k, 'pred'] + str(mention)

    output_filename = output_dir / f"{section_path.stem}_processed.tsv"
    output_df.to_csv(output_filename, sep='\t')
    print(f"Saved annotations to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM-based annotation on TSV text sections.")
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='List of input TSV files to process.' )
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output TSV and debug JSON files.' )
    parser.add_argument('--model', type=str, required=True, help='LLM Model string (e.g., "openai/gpt-4-turbo", "anthropic/claude-3-opus").')
    parser.add_argument('--prompt_type', type=str, default="default", help=f'Key for the prompt class to use. Options: {list(PROMPT_REGISTRY.keys())}' )
    parser.add_argument('-X', '--generation_args', type=str, required=False, action='append', help='Generation arguments.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generation_args:
        gen_args = OmegaConf.from_dotlist(args.generation_args or [])
    else:
        gen_args = OmegaConf.create(dict(
            seed=123,
            temperature=0,
            reasoning=dict(enabled=False)
        ))
        print('Picked up the following generation arguments:')
        print(OmegaConf.to_yaml(gen_args))

    PromptClass = PROMPT_REGISTRY.get(args.prompt_type, None)
    if PromptClass is None:
        print(f"Invalid prompt type: {args.prompt_type}. Options: {list(PROMPT_REGISTRY.keys())}")
        sys.exit(1)

    prompt_instance = PromptClass()
    annotator = LLMMentionAnnotator(
        model=args.model,
        prompt=prompt_instance,
        request_args=gen_args.as_dict()
    )

    for input_path_str in args.input_files:
        path = Path(input_path_str)
        annotate_section(path, annotator, prompt_instance, output_dir)


if __name__ == "__main__":
    main()