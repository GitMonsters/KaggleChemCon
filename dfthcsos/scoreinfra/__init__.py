"""Shim for dfthcsos.scoreinfra -> verifier.dfthc_math."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'starter_kit'))
sys.path.insert(0, os.path.join(ROOT, 'starter_kit', 'verifier'))
from verifier import dfthc_math as _dm
for _attr in dir(_dm):
    if not _attr.startswith('__'):
        globals()[_attr] = getattr(_dm, _attr)
from verifier import circuits as _vc
import types
circuits = types.ModuleType('dfthcsos.scoreinfra.circuits')
for _attr in dir(_vc):
    if not _attr.startswith('__'):
        setattr(circuits, _attr, getattr(_vc, _attr))
sys.modules['dfthcsos.scoreinfra.circuits'] = circuits