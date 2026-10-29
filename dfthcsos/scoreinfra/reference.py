"""Shim for dfthcsos.scoreinfra.reference -> verifier.reference."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'starter_kit'))
sys.path.insert(0, os.path.join(ROOT, 'starter_kit', 'verifier'))
from verifier import reference as _vr
for _attr in dir(_vr):
    if not _attr.startswith('__'):
        globals()[_attr] = getattr(_vr, _attr)

# Add missing attributes
UPPER_ENERGY = -22140.27335

def check_fcidump():
    return _vr.fcidump_path('FeMoCo76o113e')