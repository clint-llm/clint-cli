import argparse
import json
from pathlib import Path
from typing import *

import numpy
from tqdm import tqdm

from ..utils import build_document, embed_texts, openai_call_retry


class EmbedItem(NamedTuple):
    text: str
    path: Path


class EmbeddedItem(NamedTuple):
    embed: EmbedItem
    embedding: numpy.ndarray


def embed_batch(batch: List[EmbedItem]) -> List[EmbeddedItem]:
    embeddings: Optional[numpy.ndarray] = openai_call_retry(
        lambda: embed_texts([x for x, _ in batch])
    )
    if embeddings is not None:
        return [
            EmbeddedItem(embed, embedding)
            for embed, embedding in zip(batch, embeddings)
        ]

    embedded = []
    for embed in batch:
        embeddings = openai_call_retry(lambda: embed_texts([embed.text]))
        if embeddings is None:
            continue
        embedded.append(EmbeddedItem(embed, embeddings[0]))
    return embedded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    parts: List[Path] = []
    paths: List[Path] = [args.path]
    while paths:
        path = paths.pop()
        if path.is_dir() and path.suffix == ".parts":
            paths += [x for x in path.iterdir()]
        if not path.is_file() or not path.suffixes == [".meta", ".json"]:
            continue
        parts.append(path.with_suffix("").with_suffix(""))

    batch_size = 8  # hits the rate limit sometimes
    batch: List[EmbedItem] = []

    for part_path in tqdm(parts):
        meta = json.load(part_path.with_suffix(".meta.json").open("r"))
        content = meta.get("content")
        if not content:
            continue

        path_embedding = part_path.with_suffix(".npy")
        if path_embedding.exists():
            continue

        # include title in the embedding to capture more context
        document = build_document(part_path, skip_title=False)
        batch.append(EmbedItem(document, path_embedding))

        if len(batch) < batch_size:
            continue

        for item in embed_batch(batch):
            numpy.save(str(item.embed.path), numpy.array(item.embedding))
        batch = []

    if batch:
        for item in embed_batch(batch):
            numpy.save(str(item.embed.path), numpy.array(item.embedding))


if __name__ == "__main__":
    main()
