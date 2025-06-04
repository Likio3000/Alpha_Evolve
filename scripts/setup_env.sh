#!/bin/sh
# Simple environment setup script
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
