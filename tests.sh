#!/usr/bin/env bash
# set -euo pipefail

# # 1) Create & activate a venv
# python3 -m venv .venv_tests
# source .venv_tests/bin/activate

# 2) Bootstrap the Python project
pip install --upgrade pip
pip install -e .

# 3) Run Rust unit tests
cargo test

# 4) Build & install the Python extension in release mode
maturin develop --release

# 5) Run the Python test suite
pytest --maxfail=1 --disable-warnings -v

# 6) Cleanup
# deactivate
# rm -rf .venv_tests
# cargo clean
