'''
Shared pickle loading helper used by all eval/training loaders.

Centralizes the "does the file exist + can we read it" logic so that the
type-specific loaders (load_series, load_dataset, ...) only have to decide
the failure mode (strict sys.exit vs optional None).
'''
import os
import pickle
from typing import Optional

from utils.log import get_logger
logger = get_logger(__name__)

def safe_load_pickle(path: str, label: str) -> Optional[object]:
    '''
    Load a pickle file. Returns None on missing or unreadable file (logging
    a warning), instead of raising. The caller decides whether None is fatal.
    '''
    if not os.path.exists(path):
        logger.warning(f"{label} not found at {path}; skipping.")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load {label} at {path}: {e}; skipping.")
        return None
