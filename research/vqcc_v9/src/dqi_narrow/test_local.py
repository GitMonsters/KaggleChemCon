"""Round trip of the local in-place inverse: inverse24_forward190 then original vs narrow reverse."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'exp/decoder'))
import kickmix as km
from exp.decoder.common import K, write, read
from exp.inverse12_escape.joint import inverse24_forward190, inverse24_reverse209
from exp.narrow.kernel import inverse24_reverse_narrow
shots = 1024
rng = np.random.default_rng(27); xv = rng.integers(0, 1 << 24, shots); xv[:16] = 0; xv[16:24] = 1
for name, rev in [('orig', inverse24_reverse209), ('narrow', inverse24_reverse_narrow)]:
    b = km.CircuitBuilder(); x = b.alloc_qubits(24)
    m = inverse24_forward190(b, x, K); c1 = b.finish_circuit(); T1 = c1.operation_statistics().num_toffoli_like
    rev(b, x, m, K); c = b.finish_circuit()
    sim = km.Simulator(batch_size=shots, seed=3); sim.use_same_registers_as(c); write(sim, x, xv); sim.do(c)
    junk = km.array([km.q(i) for i in range(24, c.num_qubits)])
    print(name, json.dumps(dict(fwd_T=T1, rev_T=c.operation_statistics().num_toffoli_like - T1, Q=c.num_qubits,
          restored=bool((read(sim, x) == xv).all()), scratch=bool(sim.read_across_shots(junk).any()), phase=int(sim.read_phase_flipped_across_shots().sum()))), flush=True)
