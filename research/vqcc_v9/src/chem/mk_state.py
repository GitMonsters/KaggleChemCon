"""Write a (K, W) state npz from a submission npz (two-body signed words + W)."""
import sys
import numpy as np
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
z = L.load_submission(sys.argv[1])
R, B, C = z['W'].shape
sw = L.signed_words(z['angle_words'], int(z['brot']))
np.savez(sys.argv[2], K=sw[76:].reshape(R, B, 75), W=z['W'])
print('wrote', sys.argv[2], (R, B, C))
