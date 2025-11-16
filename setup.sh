#!/usr/bin/env bash
set -e

python3 -m venv .venv

# Use the venv's pip directly
.venv/bin/pip install -r requirements.txt

