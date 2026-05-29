from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from llama2_HeavySimplex_inference_common import Llama2SimplexWaterLFQA
