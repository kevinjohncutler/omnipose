# Minimal Sphinx config for omnirefactor docs tests.
import os
import sys

project = "omnirefactor"
author = "Omnipose"
version = "0"
release = version

extensions = []
templates_path = []
exclude_patterns = []
master_doc = "index"

conf_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(conf_dir, "..", "src")))
