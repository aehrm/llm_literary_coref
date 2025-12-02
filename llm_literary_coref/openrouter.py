import json
import logging
import os
import shutil
import sys

import requests

logger = logging.getLogger(__name__)

# this dict provides sensible default providers for some specified models, in order
# to ensure full determinism and reproducibility.
OPENROUTER_PROVIDER = {
    "google/gemini-2.5-flash-lite": "google-vertex",
    "google/gemini-2.5-flash": "google-vertex",
    "qwen/qwen3-30b-a3b-instruct-2507": "chutes/bf16",
    "deepseek/deepseek-v3.2-exp": "novita",
}

class StreamingOutputHandler:
    def __init__(self, height: int = 8, prefix: str = 'LLM'):
        self.full_response = []
        self.prefix = prefix
        self.num_tokens = 0
        self.height = height
        self.term_width = shutil.get_terminal_size().columns

        sys.stderr.write("\n" * self.height)
        sys.stderr.flush()

    def handle_token(self, token: str):
        self.num_tokens += 1
        self.full_response.append(token)
        self._update_display()

    def _update_display(self):
        text = ''.join(self.full_response)
        columns, _ = shutil.get_terminal_size()
        wrapped_lines = []
        for line in text.splitlines():
            if not line:
                wrapped_lines.append("")
                continue
            wrapped_lines.append(line[:columns])

        if text.endswith('\n'):
            wrapped_lines.append("")

        content_height = self.height - 1
        visible_lines = wrapped_lines[-content_height:]

        while len(visible_lines) < content_height:
            visible_lines.insert(0, "")

        header = f"\033[94m--- {self.prefix} Stream ({self.num_tokens} toks) ---\033[0m"

        sys.stderr.write("\033[s")  # Save current cursor position
        sys.stderr.write(f"\033[{self.height}A")  # Move cursor UP 'height' lines

        sys.stderr.write(f"\r\033[K{header}\n")

        for line in visible_lines:
            sys.stderr.write(f"\r\033[K\033[90m{line}\033[0m\n")

        sys.stderr.write("\033[u")
        sys.stderr.flush()

    def get_complete_response(self) -> str:
        return ''.join(self.full_response)

    def cleanup(self):
        sys.stderr.write("\033[s")  # Save pos
        sys.stderr.write(f"\033[{self.height}A")  # Move up
        sys.stderr.write("\033[J")  # Clear everything below cursor
        sys.stderr.write("\033[u")  # Restore pos
        sys.stderr.write(f"\033[{self.height}A")
        sys.stderr.flush()

def _streaming_openrouter(request_data, prefix='', API_KEY=None):
    API_KEY = API_KEY if API_KEY is not None else os.getenv('OPENROUTER_API_KEY')
    handler = StreamingOutputHandler(prefix=prefix)
    request_id = None
    usage = None
    reason = None
    request_output = b""
    buffer = b""
    with requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
            },
            data=json.dumps(request_data),
            stream=True,
    ) as r:
        if r.status_code != 200:
            raise Exception(r.content)

        for chunk in r.iter_content(chunk_size=1024, decode_unicode=False):
            buffer += chunk
            request_output += chunk
            while True:
                line_end = buffer.find(b'\n')
                if line_end == -1:
                    break
                line = buffer[:line_end].decode('utf-8', errors='replace').strip()
                buffer = buffer[line_end + 1:]
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    data_obj = json.loads(data)
                    content = data_obj["choices"][0]["delta"].get("content")
                    if content:
                        handler.handle_token(content)

                    if 'usage' in data_obj.keys():
                        usage = data_obj['usage']

                    if 'id' in data_obj.keys():
                        request_id = data_obj['id']

                    if data_obj['choices'][0]['finish_reason'] is not None:
                        reason = data_obj['choices'][0]['finish_reason']
    handler.cleanup()

    if reason is None or usage is None:
        logger.warning('message ended prematurely')
    else:
        logger.info(f"{request_data['model']}: {usage['prompt_tokens']} in -> {usage['completion_tokens']} out")

    return handler.get_complete_response(), {'usage': usage, 'finish_reason': reason, 'request_id': request_id}

def make_openrouter_request(user_prompt: str, model: str, request_args: any, max_attempts=5, API_KEY=None):
    data = {
        "model": model,
        "stream": True,
        **request_args,
    }

    metas = []
    full_response = ""
    n_attempts = 0
    while n_attempts < max_attempts:
        logger.info(f'Prompt attempt #{n_attempts + 1} to {model}')
        if full_response:
            continuation_prompt = (
                "Your previous response was cut off due to token limits. "
                "Please continue where you left off."
            )
            messages= [ { "role": "user", "content": user_prompt },
                        { "role": "assistant", "content": full_response },
                        { "role": "user", "content": continuation_prompt }]
        else:
            messages = [ { "role": "user", "content": user_prompt } ]

        response, meta = _streaming_openrouter({'messages': messages, **data}, prefix=f'Attempt #{n_attempts+1}', API_KEY=API_KEY)
        metas.append(meta)
        full_response += response[:response.rfind('\n')+1] if '\n' in response else response
        finish_reason = meta['finish_reason']
        if finish_reason == "stop":
            break
        elif finish_reason == "length":
            n_attempts += 1
            logger.info('restarting due to length abort')
        else:
            n_attempts += 1
            logger.info('restarting due to premature finish')

    return full_response, metas
