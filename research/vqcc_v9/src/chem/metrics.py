"""Official dfthcsos metrics of an npz's own angle_words (no circuit simulation)."""
import json
import sys
sys.path[:0] = ['/home/omen/vqcwork/comp/verifiers', '/home/omen/vqcwork/comp/kickmix']
import numpy as np
from dfthcsos import verify as V
for path in sys.argv[1:]:
    z = np.load(path)
    inputs = V.load_inputs(path)
    m = V.evaluate_words(inputs, z['angle_words'], z['tree_edges'], int(z['brot']))
    keys = ('error_1', 'error_2', 'Lambda', 'E_gap', 'lambda_eff')
    print(path, json.dumps({k: float(m[k]) for k in keys if k in m}))
