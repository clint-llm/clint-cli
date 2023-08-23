import argparse
import gzip
import json
import warnings
from pathlib import Path
from typing import *

import numpy
from tqdm import tqdm

from ..utils import build_document, build_hash

README = """\
The copyright notices for the documents in the `documents/` directory can be found in the `copyrights.csv.gz` file.
The license terms for the documents in the `documents/` directory can be found in the `licenses.csv.gz` file.
Each line in the file `copyrights.csv`/`licenses.csv` files starts with a document's file name,
followed by a tab character and the copyright/license that applies to the document.\
"""


class CachedStrings:
    def __init__(self):
        self.strings: Dict[Optional[str], Optional[str]] = {}

    def __getitem__(self, key: Optional[str]) -> Optional[str]:
        if key is None:
            return None
        value = self.strings.get(key)
        if value is not None:
            return value
        self.strings[key] = key
        return self.strings.get(key)


CACHED_STRINGS = CachedStrings()


class DocumentInfo(NamedTuple):
    hash: bytes
    title: Optional[str]
    embedding: Optional[numpy.ndarray]
    url: Optional[str]
    path: Path
    copyright: Optional[str]
    license: Optional[str]


def build_pca_mapping(data: numpy.ndarray, n_components: int) -> numpy.ndarray:
    # Compute the mean of the data
    mean = numpy.mean(data, axis=0)
    # Center the data
    centered_data = data - mean
    # Compute the covariance matrix
    covariance_matrix = numpy.cov(centered_data, rowvar=False)
    # Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance_matrix)
    # Sort the eigenvalues and eigenvectors in descending order
    sort_idx = numpy.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_idx]
    # Select the top n_components eigenvectors
    eigenvectors = eigenvectors[:, :n_components]
    return eigenvectors


def test_pca_roundtrip() -> None:
    rng = numpy.random.default_rng(123)
    data = rng.standard_normal((100, 10))
    mapping = build_pca_mapping(data, n_components=9)
    decomposed = numpy.dot(data, mapping)
    recomposed = numpy.dot(decomposed, mapping.T)
    # about 1/2 of the time the values should be within 1/10
    assert numpy.mean((1 - numpy.abs(data / recomposed)) < 1e-1) > 0.4


test_pca_roundtrip()


def store_document(hash: bytes, document: str, path: Path, n_levels: int):
    hash_hex = hash.hex()
    for c in hash_hex[:n_levels]:
        path = path / c
        path.mkdir(parents=True, exist_ok=True)
    with (path / hash_hex).with_suffix(".md").open("w") as fio:
        fio.write(document)


HASH_CACHE: Dict[Path, bytes] = {}


def build_part_info(path: Path) -> Optional[DocumentInfo]:
    hash = HASH_CACHE.get(path)
    if hash is None:
        hash = build_hash(build_document(path, skip_title=True))
    HASH_CACHE[path] = hash

    embedding = None
    try:
        embedding = numpy.load(path.with_suffix(".npy"))
    except FileNotFoundError:
        pass

    url = None
    title = None
    copyright = None
    license = None
    try:
        meta = json.load(path.with_suffix(".meta.json").open("r"))
        url = meta.get("url")
        title = meta.get("title")
        copyright = CACHED_STRINGS[meta.get("copyright")]
        license = CACHED_STRINGS[meta.get("license")]
    except (FileNotFoundError, AttributeError):
        pass
    return DocumentInfo(hash, title, embedding, url, path, copyright, license)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_parts", type=Path)
    parser.add_argument("path_db", type=Path)
    parser.add_argument("--skip_parts", nargs="*", type=Path)
    args = parser.parse_args()

    path_db: Path = args.path_db
    path_db.mkdir(parents=True, exist_ok=True)

    skip_parts: Set[Path] = set(args.skip_parts)

    parts: List[Path] = []
    paths: List[Path] = [args.path_parts]
    while paths:
        path = paths.pop()
        if path.is_dir() and path.suffix == ".parts":
            paths += [x for x in path.iterdir()]
        if not path.is_file() or not path.suffixes == [".meta", ".json"]:
            continue
        parts.append(path.with_suffix("").with_suffix(""))

    path_to_info: Dict[Path, Optional[DocumentInfo]] = {}

    seen_hashes: Set[bytes] = set()
    embeddings: List[numpy.ndarray] = []
    embeddings_hash: List[bytes] = []
    titles: Dict[bytes, str] = {}
    parents: Dict[bytes, bytes] = {}
    urls: Dict[bytes, str] = {}
    copyrights: Dict[bytes, str] = {}
    licenses: Dict[bytes, str] = {}
    is_condition: Set[bytes] = set()
    is_introduction: Set[bytes] = set()
    is_symptoms: Set[bytes] = set()

    for part_path in tqdm(parts):
        if part_path.with_suffix(".parts") in skip_parts:
            continue
        info = path_to_info.get(part_path, build_part_info(part_path))
        path_to_info[part_path] = info
        if info is None:
            continue

        parent_path = (
            part_path.parent.with_suffix("")
            if part_path.parent.suffix == ".parts"
            and part_path.parent not in skip_parts
            else None
        )
        parent_info = (
            path_to_info.get(parent_path, build_part_info(parent_path))
            if parent_path is not None
            else None
        )
        if parent_path is not None:
            path_to_info[parent_path] = parent_info
        if parent_info is not None:
            parents[info.hash] = parent_info.hash

    for info in tqdm(path_to_info.values()):
        if info is None or info.hash in seen_hashes:
            continue
        seen_hashes.add(info.hash)
        if info.title is None:
            warnings.warn(f"Document at {info.path} has no title.")
            continue
        store_document(
            info.hash,
            build_document(info.path, skip_title=True),
            path_db / "documents",
            3,
        )
        if info.embedding is not None:
            embeddings.append(info.embedding)
            embeddings_hash.append(info.hash)
        titles[info.hash] = info.title
        if info.url:
            urls[info.hash] = info.url
        if info.copyright:
            copyrights[info.hash] = info.copyright
        if info.license:
            licenses[info.hash] = info.license
        if info.title == "History and Physical":
            # this section describes symptoms of a condition
            is_symptoms.add(info.hash)
            # the parent article describes a condition
            if parent_hash := parents.get(info.hash):
                is_condition.add(parent_hash)
        if info.title == "Introduction":
            # this section is an introduction
            is_introduction.add(info.hash)

    for hash in seen_hashes:
        if hash in copyrights:
            continue
        parent = parents.get(hash)
        while parent is not None:
            copyright = copyrights.get(parent)
            if copyright is None:
                parent = parents.get(parent)
                continue
            copyrights[hash] = copyright
            break

    for hash in seen_hashes:
        if hash in licenses:
            continue
        parent = parents.get(hash)
        while parent is not None:
            license = licenses.get(parent)
            if license is None:
                parent = parents.get(parent)
                continue
            licenses[hash] = license
            break

    if embeddings:
        embeddings_array = numpy.vstack(embeddings)
        pca_mapping = build_pca_mapping(embeddings_array, 128)
        numpy.save(
            str(path_db / "embeddings_pca_128_mapping.npy"),
            pca_mapping.astype(numpy.float32),
        )
        numpy.save(
            str(path_db / "embeddings_pca_128.npy"),
            numpy.dot(embeddings_array, pca_mapping).astype(numpy.float32),
        )
        with (path_db / "embeddings_hash.csv").open("w") as fio:
            for hash in embeddings_hash:
                fio.write(f"{hash.hex()}\n")

    with (path_db / "parents.csv").open("w") as fio:
        for hash, parent in sorted(parents.items()):
            fio.write(f"{hash.hex()}\t{parent.hex()}\n")

    with (path_db / "titles.csv").open("w") as fio:
        for hash, title in sorted(titles.items()):
            fio.write("{}\t{}\n".format(hash.hex(), title.replace("\t", " ")))

    with (path_db / "urls.csv").open("w") as fio:
        for hash, url in sorted(urls.items()):
            fio.write("{}\t{}\n".format(hash.hex(), url.replace("\t", "%09")))

    with gzip.open(path_db / "copyrights.csv.gz", "wb") as fio:
        for hash, copyright in sorted(copyrights.items()):
            fio.write(
                "{}\t{}\n".format(hash.hex(), copyright.replace("\t", " ")).encode(
                    "utf8"
                )
            )

    with gzip.open(path_db / "licenses.csv.gz", "wb") as fio:
        for hash, license in sorted(licenses.items()):
            fio.write(
                "{}\t{}\n".format(hash.hex(), license.replace("\t", " ")).encode("utf8")
            )

    with (path_db / "is_condition.csv").open("w") as fio:
        for hash in sorted(is_condition):
            fio.write(f"{hash.hex()}\n")

    with (path_db / "is_introduction.csv").open("w") as fio:
        for hash in sorted(is_introduction):
            fio.write(f"{hash.hex()}\n")

    with (path_db / "is_symptoms.csv").open("w") as fio:
        for hash in sorted(is_symptoms):
            fio.write(f"{hash.hex()}\n")

    with (path_db / "README.md").open("w") as fio:
        fio.write(README)


if __name__ == "__main__":
    main()
