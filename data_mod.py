import os
from pathlib import Path
import sys
SCRIPT_DIR =Path(__file__).parents[0]
print(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from args import parse_args

arg=parse_args