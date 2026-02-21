"""
pytest configuration – ensures the project root is on sys.path so that
absolute imports (``from flow import ...``, ``import openrouter``, etc.) work
when running tests from any directory.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
