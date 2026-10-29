import numpy as np
import hashlib
import json
from pathlib import Path

# Load the orthogonal gauge
O = np.load('orthogonal_gauge.npy')
print('Gauge shape:', O.shape)

# Load the busyaprime submission
sub_path = Path('submission_busyaprime/dfthcsos/submission.npz')
if not sub_path.exists():
    # Try from the extracted busyaprime members
    sub_path = Path('research/busyaprime_members/dfthcsos__submission.npz')

data = np.load(sub_path, allow_pickle=False)
print('Keys:', list(data.keys()))
for k in data.keys():
    arr = data[k]
    print(f'  {k}: shape={arr.shape}, dtype={arr.dtype}')

# Apply gauge
W = np.array(data['W'], dtype=np.float64)
H2_I = np.array(data['H2_I'], dtype=np.float64)
print('W shape:', W.shape)
print('H2_I shape:', H2_I.shape)

# Concatenate and rotate
A = np.concatenate([W, H2_I[:, None, :]], axis=1)  # (R, B+1, 128)
A1 = A @ O  # (R, B+1, 128)

# Split back
W_new = A1[:, :-1, :]
H2_I_new = A1[:, -1, :]
offset_W_new = H2_I_new - W_new.sum(axis=1)

# Verify Gram matrix preservation
gram_before = A @ A.swapaxes(1, 2)
gram_after = A1 @ A1.swapaxes(1, 2)
print('Gram matrix max diff:', np.max(np.abs(gram_before - gram_after)))

# Save
new_data = {k: np.array(data[k]) for k in data.keys()}
new_data['W'] = W_new.astype(data['W'].dtype)
new_data['H2_I'] = H2_I_new.astype(data['H2_I'].dtype)
if 'offset_W' in new_data:
    new_data['offset_W'] = offset_W_new.astype(data['offset_W'].dtype)

# Save the polished submission
out_dir = Path('submission_polished/dfthcsos')
out_dir.mkdir(parents=True, exist_ok=True)
np.savez(out_dir / 'submission.npz', **new_data)

# Copy other files
import shutil
src_dir = Path('submission_busyaprime')
dst_dir = Path('submission_polished')
for f in src_dir.rglob('*'):
    if f.is_file() and f.name != 'submission.npz':
        rel = f.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)

print('Saved polished submission to submission_polished/')
print('Original submission.npz hash:', hashlib.sha256(sub_path.read_bytes()).hexdigest())
print('Polished submission.npz hash:', hashlib.sha256((out_dir / 'submission.npz').read_bytes()).hexdigest())
