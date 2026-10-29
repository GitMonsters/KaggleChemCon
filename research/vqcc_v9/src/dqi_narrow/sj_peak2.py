"""Exact live-qubit composition at each stage's peak (all qubit ids, not just >= 840).

Usage (cwd = session dir, PYTHONPATH = official kickmix): python sj_peak2.py <source_root> <build_script.py> [stages]
Each live qubit is tagged with (stage it was allocated in, allocation site). Qubits allocated before the first
mark (the 840 syndrome wires) are tagged '-'.
"""
import sys, runpy, traceback, collections
from pathlib import Path
import kickmix as km

root = Path(sys.argv[1]); script = root / 'blocks/dqi/team-fused/source/exp/decoder' / sys.argv[2]
want = set(sys.argv[3].split(',')) if len(sys.argv) > 3 else None
state = dict(stage='-', main=None, owner={}, peak=collections.Counter(), comp={})
Orig = km.CircuitBuilder


def site(depth=6):
    out = []
    f = sys._getframe(2)
    while f is not None and len(out) < depth:
        fn = f.f_code.co_filename
        if '/exp/' in fn:
            out.append(f'{Path(fn).parent.name}/{Path(fn).stem}.{f.f_code.co_name}:{f.f_lineno}')
        f = f.f_back
    return ' < '.join(out)


class Tracer:
    def __init__(self, *a, **kw):
        self._b = Orig(*a, **kw)
        if state['main'] is None:
            state['main'] = self

    def __getattr__(self, name):
        return getattr(self._b, name)

    def mark(self, name, *a, **kw):
        if self is state['main']:
            state['stage'] = name
        return self._b.mark(name, *a, **kw)

    def alloc_qubits(self, k, *a, **kw):
        out = self._b.alloc_qubits(k, *a, **kw)
        if self is not state['main']:
            return out
        st = state['stage']
        s = site() if st != '-' else 'syndrome'
        for q in out:
            state['owner'][q.id] = (st, s)
        live = self._b.num_allocated_qubits
        if live > state['peak'][st] and 'reclaim' not in s and (want is None or st in want):
            state['peak'][st] = live
            state['comp'][st] = collections.Counter(v for v in state['owner'].values() if v[0] != '-')
        return out

    def free(self, qs, *a, **kw):
        if self is state['main']:
            try:
                ids = [q.id for q in qs]
            except TypeError:
                ids = [qs.id]
            for i in ids:
                state['owner'].pop(i, None)
        return self._b.free(qs, *a, **kw)


km.CircuitBuilder = Tracer
sys.argv = [str(script), '/tmp/sj_peak2_dummy.kmb']
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name='__main__')
for st, comp in state['comp'].items():
    print(f'== {st}: peak {state["peak"][st]} allocated, non-syndrome live {sum(comp.values())}')
    for (ast, s), k in comp.most_common(30):
        print(f'   {k:4d}  [{ast}] {s}')
