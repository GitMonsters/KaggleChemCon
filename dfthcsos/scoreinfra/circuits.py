"""Shim for dfthcsos.scoreinfra.circuits -> verifier.circuits."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'starter_kit'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'starter_kit', 'verifier'))
from verifier import circuits as _vc
for _attr in dir(_vc):
    if not _attr.startswith('__'):
        globals()[_attr] = getattr(_vc, _attr)
"