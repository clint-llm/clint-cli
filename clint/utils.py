import hashlib
import json
import time
from pathlib import Path
from typing import *

import numpy
import openai
import openai.error
import tiktoken


def build_document(path: Path, skip_title: bool) -> str:
    contents = []
    parts_path = [path]
    first_title = True
    while parts_path:
        part_path = parts_path.pop()
        meta: Dict[str, Any] = json.load(part_path.with_suffix(".meta.json").open("r"))
        title = None if (first_title and skip_title) else meta.get("title")
        first_title = False
        if title:
            contents.append(f"# {title}")
        content_path: Optional[str] = meta.get("content")
        if content_path is not None:
            contents.append((part_path.parent / content_path).read_text())
        parts_path += [
            part_path.parent / Path(x) for x in reversed(meta.get("parts", []))
        ]
    return "\n\n".join(contents)


def build_hash(document: str) -> bytes:
    hash = hashlib.sha256(document.encode("utf8")).digest()[:16]
    return hash


def embed_texts(texts: List[str]) -> numpy.ndarray:
    encoding = tiktoken.get_encoding("cl100k_base")
    texts = [encoding.decode(encoding.encode(x)[:8191]) for x in texts]
    response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
    return numpy.array([x["embedding"] for x in response["data"]])


def openai_call_retry(call: Callable[[], Any]) -> Optional[Any]:
    backoff_base = 1.0
    backoff = backoff_base
    has_error = False

    while True:
        try:
            return call()
        except openai.error.RateLimitError:
            # backoff until back under rate limit
            print("Rate limit, sleep: {}".format(backoff))
            time.sleep(backoff)
            backoff *= 2
            continue
        except openai.error.InvalidRequestError as err:
            print("Request error: {}".format(err))
            # request won't work so move onto next
            return None
        except openai.error.OpenAIError as err:
            print("OpenAI error: {}".format(err))
            # re-try once in case of intermittent error, but then move on since
            # there could be something wrong with the request
            if has_error:
                return None
            has_error = True
            continue
