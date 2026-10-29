"""Shim to make dfthcsos.scoreinfra available as verifier.dfthc_math."""
import sys
sys.path.insert(0, '/kaggle/input/competitions/google-verifiable-quantum-circuit-compilation/verifiers')
# Create the dfthcsos package
import types
dfthcsos = types.ModuleType('dfthcsos')
sys.modules['dfthcsos'] = dfthcsos
# Create scoreinfra submodule
scoreinfra = types.ModuleType('dfthcsos.scoreinfra')
import verifier.dfthc_math as dm
for attr in dir(dm):
    if not attr.startswith('__'):
        setattr(scoreinfra, attr, getattr(dm, attr))
sys.modules['dfthcsos.scoreinfra'] = scoreinfra
dfthcsos.scoreinfra = scoreinfra
print('Shim loaded successfully')
"