"""Compare raw .npy ZIP members of protected fields between a reference npz and candidates (must be byte-identical)."""
import hashlib
import sys
import zipfile

ref = sys.argv[1]
keys = ['H1_I.npy', 'Hsym_lower.npy', 'tree_edges.npy']
with zipfile.ZipFile(ref) as z:
    want = {k: hashlib.sha256(z.read(k)).hexdigest() for k in keys}
ok = True
for cand in sys.argv[2:]:
    with zipfile.ZipFile(cand) as z:
        got = {k: hashlib.sha256(z.read(k)).hexdigest() for k in keys}
    same = all(got[k] == want[k] for k in keys)
    ok &= same
    print(cand, 'protected fields byte-identical' if same else f'MISMATCH {[k for k in keys if got[k] != want[k]]}')
sys.exit(0 if ok else 1)
