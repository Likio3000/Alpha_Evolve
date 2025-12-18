#!/bin/sh
# Simple environment setup script
set -e
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
