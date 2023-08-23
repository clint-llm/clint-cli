# Clint CLI

This is the web UI for the [Clint LLM project](https://github.com/clint-llm/clint-llm.github.io).
Check out the [live project](https://clint-llm.github.io).

Clint started as a Python prototype, 
and was eventually split into a Typescript UI and Rust library.
The only Python code remaining is for building the database resources.

## Development

### Environment

- You need Python and Poetry to use the CLI
  - <https://www.python.org/>
  - <https://python-poetry.org/>

### Initial build setup

The project was initialized with the following commands (included for documentation):

- Poetry for setting up the CLI tools:
  - `poetry new clint-cli`

## Usage

```bash
mkdir data/ && cd data/
curl https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz --output statpearls_NBK430685.tar.gz
tar -xzf statpearls_NBK430685.tar.gz
poetry install
poetry shell
clint-build-statpearls data/statpearls_NBK430685 data/docs
clint-build-embeddings data/docs/StatPearls.parts
clint-build-db data/docs/StatPearls.parts/ data/db v1 --skip_parts data/docs/StatPearls.parts
```

