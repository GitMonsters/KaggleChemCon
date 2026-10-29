"""Static Toffoli profile of the SJ fused DQI build by stage and SJ-source call chain.

Counts builder-level Toffoli emitters (ccx, ccz, cswap, init_and, multi_controlled_x controls-1).
Usage (cwd = session dir, PYTHONPATH = official kickmix): python sj_tprofile.py <source_root> [depth]
"""
import sys, runpy, traceback, collections
from pathlib import Path
import kickmix as km

root = Path(sys.argv[1]); depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
script = root / 'blocks/dqi/team-fused/source/exp/decoder/build.py'
state = dict(stage='-', main=None)
cnt = collections.Counter(); stage_tot = collections.Counter()
Orig = km.CircuitBuilder


_names = {}


def where():
    f = sys._getframe(2); chain = []
    while f is not None:
        co = f.f_code; fn = co.co_filename
        if '/exp/' in fn and 'decoder/build' not in fn:
            key = (fn, co.co_name)
            nm = _names.get(key)
            if nm is None:
                pth = Path(fn); nm = _names[key] = f'{pth.parent.name}/{pth.stem}.{co.co_name}'
            chain.append(nm)
        f = f.f_back
    chain.reverse()
    return ' > '.join(chain[:depth])


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

    def _count(self, k):
        if self is state['main'] and k:
            st = state['stage']; cnt[(st, where())] += k; stage_tot[st] += k

    def _q(self, a):
        n = 0
        for x in a:
            if isinstance(x, km.array):
                if len(x) and isinstance(x[0], km.b):
                    return 0
                n = max(n, len(x))
            elif isinstance(x, km.b):
                return 0
        return max(n, 1)

    def ccx(self, *a, **kw):
        self._count(self._q(a)); return self._b.ccx(*a, **kw)

    def ccz(self, *a, **kw):
        self._count(self._q(a)); return self._b.ccz(*a, **kw)

    def cswap(self, *a, **kw):
        self._count(self._q(a)); return self._b.cswap(*a, **kw)

    def init_and(self, *a, **kw):
        self._count(self._q(a)); return self._b.init_and(*a, **kw)


km.CircuitBuilder = Tracer
sys.argv = [str(script), '/tmp/sj_tprof_dummy.kmb']
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name='__main__')
import pickle; pickle.dump(dict(cnt), open('/home/omen/vqcwork/sj1027/tprof_cnt.pkl','wb'))
print('stage totals (python-counted emitters):', dict(stage_tot))
for st in stage_tot:
    print('==', st, stage_tot[st])
    for (s, w), k in sorted(((k_, v) for k_, v in cnt.items() if k_[0] == st), key=lambda x: -x[1])[:14]:
        print(f'   {k:7d}  {w}')
