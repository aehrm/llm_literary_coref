import json
import logging
import re
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional

from llm_literay_coref.openrouter import make_openrouter_request
from llm_literay_coref.prompts.prompt import Prompt

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class LLMOutput(Generic[T]):
    model: str
    raw_input: str
    raw_output: str
    generation_details: any
    json_output: Optional[list] = None
    output: Optional[T] = None
    exception: Optional[Exception] = None


def parse_json(output_str) -> list:
    parsed_output_lines = []
    for line in output_str.split('\n'):
        if line.strip() == '':
            continue
        if m := re.search(r'(\{.*\})', line):
            try:
                parsed = json.loads(m.group(1))
                parsed_output_lines.append(parsed)
            except Exception as e:
                logger.warning(f'Could not parse line {repr(line)}: {str(e)}')
        else:
            logger.warning(f'Could not parse line {repr(line)}')

    return parsed_output_lines


class LLMRunner(Generic[T]):

    def __init__(self, model: str, request_args: dict = None):
        self.model = model
        self.request_args = request_args if request_args is not None else {}

    def run(self, prompt: Prompt[T]) -> LLMOutput[T]:
        prompt_str = prompt.format_prompt()

        output_str = None
        generation_details = None
        try:
            logger.info(f"Making API request to model: {self.model}")
            output_str, generation_details = make_openrouter_request(prompt_str, self.model, self.request_args)
            logger.info(f"Successfully received response from model: {self.model}")
            json_output = parse_json(output_str)
            output = prompt.decode(json_output)

            return LLMOutput(output=output, model=self.model, raw_input=prompt_str, raw_output=output_str, json_output=json_output,
                             generation_details=generation_details)
        except Exception as e:
            logger.exception(f"Error during model inference")
            return LLMOutput(model=self.model, raw_input=prompt_str, raw_output=output_str, generation_details=generation_details,
                             exception=e)
