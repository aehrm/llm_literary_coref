import json
import logging
import re
from dataclasses import dataclass
from typing import List, TypeVar, Generic

import pandas

from llm_literay_coref.mention import Mention
from llm_literay_coref.openrouter import make_openrouter_request
from llm_literay_coref.prompts import Prompt, LLMOutput, LLMOutputLine

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LLMAnnotator(Generic[T]):

    def __init__(self, model: str, prompt: 'Prompt[T]', request_args=None):
        self.model = model
        self.request_args = request_args
        self.prompt = prompt
        self.mention_annotations = []

    def prepare_input(self, tokens: pandas.Series, mention_spans: List[Mention]):
        input_text_df = tokens.copy()

        mention_span_map = {}
        mention_annotations = []

        index_mapper = pandas.Series(range(len(tokens)), index=input_text_df.index)
        for id_, mention_span in enumerate(sorted(mention_spans, key=lambda x: x.token_idx[0])):
            begin = min(mention_span.token_idx)
            end = max(mention_span.token_idx)
            input_text_df.loc[begin] = '[' + input_text_df.loc[begin]
            input_text_df.loc[end] = str(input_text_df.loc[end]) + f'][{id_ + 1}]'
            pos = index_mapper.loc[begin]

            mention_span_map[id_ + 1] = mention_span
            mention_annotations.append({
                "ID": id_ + 1,
                "Position": int(pos),
                "Text": " ".join(tokens.loc[begin:end]),
                "Annotation": [],
            })

        formatted_input = ' '.join(input_text_df)

        return formatted_input, mention_span_map, mention_annotations

    def run(self, tokens: pandas.Series, mention_spans: List[Mention]) -> LLMOutput[T]:
        formatted_input, mention_span_map, mention_annotations = self.prepare_input(tokens, mention_spans)
        prompt = self.prompt.format_prompt(formatted_input, mention_annotations)
        request_args = {
            "reasoning": {
                "enabled": False,
            },
            **(self.request_args if self.request_args is not None else {})
        }

        ret = None
        generation_details = None
        try:
            logger.info(f"Making API request to model: {self.model}")
            ret, generation_details = make_openrouter_request(prompt, self.model, request_args)
            logger.info(f"Successfully received response from model: {self.model}")
            parsed = self.parse_result(ret, mention_span_map)

            return LLMOutput(output=parsed, model=self.model, raw_input=prompt, raw_output=ret,
                             generation_details=generation_details)
        except Exception as e:
            logger.exception(f"Error during model inference")
            return LLMOutput(model=self.model, raw_input=prompt, raw_output=ret, generation_details=generation_details,
                             exception=e)

    def parse_result(self, result_string, mention_span_map: dict[int, Mention]) -> List[LLMOutputLine[T]]:
        annotated_mentions = []

        for line in result_string.split('\n'):
            if line.strip() == '':
                continue
            if m := re.search(r'(\{.*\})', line):
                try:
                    parsed = json.loads(m.group(1))
                    id_ = parsed['ID']

                    annotation_text = parsed['Annotation']
                    annotation = self.prompt.parse_response_annotation(annotation_text)


                    annotated_span = mention_span_map[id_]
                    annotated_mentions.append(LLMOutputLine(
                        annotation=annotation,
                        mention=annotated_span
                    ))
                except Exception as e:
                    logger.warning(f'Could not parse line {repr(line)}: {str(e)}')
            else:
                logger.warning(f'Could not parse line {repr(line)}')


        return annotated_mentions
