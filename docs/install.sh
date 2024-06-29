#!/bin/bash

# Install documentation dependencies from requirements.txt
pip install -r docs/requirements.txt

# Install the local package without dependencies
pip install -e . --no-deps