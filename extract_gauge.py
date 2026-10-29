import re, base64, zlib, numpy as np, io

src = open('research/qrom_tower/quantum-circuits-qrom-tower42.py').read()
m = re.search(r"ORTHOGONAL_B64='([^']+)'", src)
if m:
    b64 = m.group(1)
    O = np.load(io.BytesIO(zlib.decompress(base64.b64decode(b64))))
    print('Shape:', O.shape)
    print('Orthogonality check:', np.max(np.abs(O.swapaxes(1,2)@O - np.eye(128))))
    np.save('orthogonal_gauge.npy', O)
    print('Saved to orthogonal_gauge.npy')
else:
    print('ORTHOGONAL_B64 not found')
    # Try to find it
    idx = src.find('ORTHOGONAL_B64')
    print('Index:', idx)
    print(src[idx:idx+100])
