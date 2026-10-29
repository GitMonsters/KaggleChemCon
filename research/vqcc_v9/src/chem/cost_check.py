"""Official circuit resources and metrics of a package even when chemistry fails (no rejection)."""
import json
import sys
from pathlib import Path
sys.path[:0] = ['/home/omen/vqcwork/comp/verifiers', '/home/omen/vqcwork/comp/kickmix']
from dfthcsos import verify as V
npz, circ = Path(sys.argv[1]), Path(sys.argv[2])
edges = V.load_tree_edges(npz, circ)
inputs = V.load_inputs(npz)
family = V.load_circuits(circ, format='auto')
res = V.evaluate_circuits(inputs, family, edges, salt='')
r, m = res['resources'], res['metrics']
print(json.dumps({'resources': {k: (float(v) if not isinstance(v, (dict, list)) else v) for k, v in r.items()},
                  'metrics': {k: float(m[k]) for k in ('error_1', 'error_2', 'Lambda', 'E_gap', 'lambda_eff')},
                  'num_circuits': len(family)}, indent=1, default=str))
lam = float(sys.argv[3]) if len(sys.argv) > 3 else 47.4
cost = lam * float(r['toffoli']) * int(r['qubits'])
print(f'ratio at lambda {lam}: {3355599588.3155365 / cost:.4f}')
