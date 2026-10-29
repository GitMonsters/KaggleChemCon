import sys
import numpy as np
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
z = L.load_submission(sys.argv[1])
brot = int(z['brot']); R, B, C = z['W'].shape
sw = L.signed_words(z['angle_words'], brot)
two = sw[76:].reshape(R, B, 75); one = sw[:76]
print('two-body words: zero frac', (two == 0).mean().round(4), ' |w|<=1 frac', (np.abs(two) <= 1).mean().round(4))
print('per-vector nonzero count quantiles', np.quantile((two != 0).sum(-1), [0, .1, .5, .9, 1]))
print('one-body zero frac', (one == 0).mean().round(4))
U = z['U'].reshape(-1, 76)
mass = np.sort(U**2, axis=1)[:, ::-1]
cum = np.cumsum(mass, axis=1)
for t in (0.9, 0.99, 0.999):
    k = (cum < t).sum(1) + 1
    print(f'orbitals for {t} mass: quantiles', np.quantile(k, [0.1, .5, .9]))
print('abs word histogram (two-body):', np.bincount(np.abs(two).ravel(), minlength=129)[:12], '...')
