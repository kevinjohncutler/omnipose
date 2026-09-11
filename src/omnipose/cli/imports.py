"""Centralized stdlib imports for the cli subpackage.

Tools-specific heavy imports (torch, numpy, tqdm) stay in the leaf that
needs them — keep this module to common stdlib so every CLI entry point
can ``from .imports import *`` without dragging in the world.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
