import json
import logging
import re
from dataclasses import dataclass
from typing import List, TypeVar, Generic, Tuple, Mapping

import pandas

from llm_literay_coref.mention import Mention, Entity
from llm_literay_coref.openrouter import make_openrouter_request
from llm_literay_coref.prompts.prompt import MentionAnnotatorOutputLine, MentionPrompt, MergePrompt, \
    MergeAnnotatorOutputLine

logger = logging.getLogger(__name__)

T = TypeVar('T')
O = TypeVar('O')


@dataclass
class LLMOutput(Generic[O]):
    model: str
    raw_input: any
    raw_output: any
    generation_details: any
    output: O = None
    exception: Exception = None

class LLMMentionAnnotator(Generic[T]):

    def __init__(self, model: str, prompt: MentionPrompt[T], request_args: dict = None):
        self.model = model
        self.request_args = request_args if request_args is not None else {}
        self.prompt = prompt

    def prepare_input(self, tokens: pandas.Series, mention_spans: List[Mention]) -> Tuple[str, dict[int, Mention], list]:
        input_text_df = tokens.copy()

        mention_span_map = {}
        span_annotations = []

        index_mapper = pandas.Series(range(len(tokens)), index=input_text_df.index)
        for id_, mention_span in enumerate(sorted(mention_spans, key=lambda x: x.token_idx[0])):
            begin = min(mention_span.token_idx)
            end = max(mention_span.token_idx)
            input_text_df.loc[begin] = '[' + input_text_df.loc[begin]
            input_text_df.loc[end] = str(input_text_df.loc[end]) + f'][{id_ + 1}]'
            pos = index_mapper.loc[begin]

            mention_span_map[id_ + 1] = mention_span
            span_annotations.append({
                "ID": id_ + 1,
                "Position": int(pos),
                "Text": " ".join(tokens.loc[begin:end]),
                "Annotation": [],
            })

        formatted_input = ' '.join(input_text_df)

        return formatted_input, mention_span_map, span_annotations

    def run(self, tokens: pandas.Series, mention_spans: List[Mention]) -> LLMOutput[List[MentionAnnotatorOutputLine[T]]]:
        formatted_input, mention_span_map, span_annotations = self.prepare_input(tokens, mention_spans)
        prompt_str = self.prompt.format_prompt(formatted_input, span_annotations)

        ret = None
        generation_details = None
        try:
            logger.info(f"Making API request to model: {self.model}")
            ret, generation_details = make_openrouter_request(prompt_str, self.model, self.request_args)
            logger.info(f"Successfully received response from model: {self.model}")
            parsed = self.parse_result(ret, mention_span_map)

            return LLMOutput(output=parsed, model=self.model, raw_input=prompt_str, raw_output=ret,
                             generation_details=generation_details)
        except Exception as e:
            logger.exception(f"Error during model inference")
            return LLMOutput(model=self.model, raw_input=prompt_str, raw_output=ret, generation_details=generation_details,
                             exception=e)

    def parse_result(self, result_string: str, mention_span_map: dict[int, Mention]) -> List[MentionAnnotatorOutputLine[T]]:
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
                    annotated_mentions.append(MentionAnnotatorOutputLine(
                        annotation=annotation,
                        mention=annotated_span
                    ))
                except Exception as e:
                    logger.warning(f'Could not parse line {repr(line)}: {str(e)}')
            else:
                logger.warning(f'Could not parse line {repr(line)}')


        return annotated_mentions


class LLMMergeAnnotator:

    def __init__(self, model: str, prompt: MergePrompt, request_args=None):
        self.model = model
        self.request_args = request_args
        self.prompt = prompt

    def prepare_input(self, tokens: pandas.Series, section_ids: pandas.Series, mentions: List[Mention]) -> list:
        # TODO group by entity and section_id and collect all non-PRON mentions
        pass

    def run(self, tokens: pandas.Series, section_ids: pandas.Series, mentions: List[Mention]) -> LLMOutput[List[MergeAnnotatorOutputLine]]:
        entity_annotations = self.prepare_input(tokens, section_ids, mentions)
        prompt_str = self.prompt.format_prompt(entity_annotations)

        ret = None
        generation_details = None
        try:
            logger.info(f"Making API request to model: {self.model}")
            ret, generation_details = make_openrouter_request(prompt_str, self.model, self.request_args)
            logger.info(f"Successfully received response from model: {self.model}")
            parsed = self.parse_result(ret)

            return LLMOutput(output=parsed, model=self.model, raw_input=prompt_str, raw_output=ret,
                             generation_details=generation_details)
        except Exception as e:
            logger.exception(f"Error during model inference")
            return LLMOutput(model=self.model, raw_input=prompt_str, raw_output=ret, generation_details=generation_details,
                             exception=e)

    def parse_result(self, result_string: str) -> List[MergeAnnotatorOutputLine]:
        annotated_entity = []

        for line in result_string.split('\n'):
            if line.strip() == '':
                continue
            if m := re.search(r'(\{.*\})', line):
                try:
                    parsed = json.loads(m.group(1))
                    annotated_entity.append(MergeAnnotatorOutputLine(
                        section_id=parsed["section_id"],
                        entity_id=parsed["entity_id"],
                        entity_id_global=parsed["entity_id_global"]
                    ))
                except Exception as e:
                    logger.warning(f'Could not parse line {repr(line)}: {str(e)}')
            else:
                logger.warning(f'Could not parse line {repr(line)}')


        return annotated_entity
