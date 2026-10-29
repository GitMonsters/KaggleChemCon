"""Shim for dfthcsos.scoreinfra.dfthc_math -> verifier.dfthc_math."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'starter_kit'))
sys.path.insert(0, os.path.join(ROOT, 'starter_kit', 'verifier'))
from verifier import dfthc_math as _dm
for _attr in dir(_dm):
    if not _attr.startswith('__'):
        globals()[_attr] = getattr(_dm, _attr)