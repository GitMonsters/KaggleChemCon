import sys, time
import numpy as np, torch
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
torch.set_num_threads(4)
z = L.load_submission(sys.argv[1])
R, B, C = z['W'].shape
brot = int(z['brot'])
tb = L.TwoBody()
tree = L.Tree(z['tree_edges'])
words = z['angle_words']
th = torch.tensor(L.theta_of_words(words, brot), device='cuda')
Uq = tree(th)                     # (76 + R*B, 76)
U2 = Uq[N0:] if (N0 := 76) else None
U2 = U2.reshape(R, B, 76)
w = torch.tensor(z['W'], device='cuda')
t0 = time.time()
e_q = tb.err(U2, w).item()
print('decoded-word U error_2 mHa', 1e3 * e_q, f'({time.time()-t0:.1f}s)')
print('npz U error_2 mHa', 1e3 * tb.err(torch.tensor(z['U'], device='cuda'), w).item())
print('max |U_npz - U_decoded| (up to sign)', float(torch.minimum((torch.tensor(z['U'],device='cuda')-U2).abs().amax(-1), (torch.tensor(z['U'],device='cuda')+U2).abs().amax(-1)).max()))
if 'continuous_U' in z:
    print('continuous_U error_2 mHa', 1e3 * tb.err(torch.tensor(z['continuous_U'], device='cuda'), w).item())
print('D(w)', L.dpen(w).item(), ' |w| mean', float(w.abs().mean()))
# weight scale per vector: omega_b = sum_c w^2
om = (w * w).sum(-1)
print('omega quantiles', np.quantile(om.cpu().numpy(), [0, .1, .5, .9, 1]).round(5))
