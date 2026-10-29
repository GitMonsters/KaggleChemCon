"""Qubit high-water profile of the SomeJoint fused DQI build (per stage, with live allocation sites).

Usage (cwd = session dir, PYTHONPATH = official kickmix):
  python sj_peak.py <source_root> [stage_filter]
Records, for every new global high-water mark of allocated qubits, the stage and the
allocation sites of all live non-syndrome qubits.
"""
import sys, runpy, traceback, collections, json
from pathlib import Path
import kickmix as km

root = Path(sys.argv[1])
script = root / 'blocks/dqi/team-fused/source/exp/decoder' / (sys.argv[2] if len(sys.argv) > 2 else 'build.py')
state = dict(stage='-', best=840, owner={}, events=[], stage_peak=collections.Counter(), main=None)
Orig = km.CircuitBuilder


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
        stack = traceback.extract_stack()[:-1]
        site = ' > '.join(f'{Path(f.filename).parent.name}/{Path(f.filename).stem}.{f.name}:{f.lineno}' for f in stack[-7:])
        for q in out:
            state['owner'][q.id] = site
        live = self._b.num_allocated_qubits
        st = state['stage']
        if live > state['stage_peak'][st]:
            state['stage_peak'][st] = live
            cnt = collections.Counter(v for qid, v in state['owner'].items() if qid >= 840)
            state.setdefault('stage_sites', {})[st] = cnt.most_common(12)
        if live > state['best'] and 'reclaim' not in site:
            state['best'] = live
            cnt = collections.Counter(v for qid, v in state['owner'].items() if qid >= 840)
            state['events'].append(dict(live=live, stage=st, sites=cnt.most_common(25)))
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
sys.argv = [str(script), '/tmp/sj_peak_dummy.kmb']
sys.path.insert(0, str(script.parent))
try:
    runpy.run_path(str(script), run_name='__main__')
except SystemExit:
    pass
print('stage peaks (allocated, incl. 840 syndrome):', dict(state['stage_peak']))
for st, sites in state.get('stage_sites', {}).items():
    print('== stage peak', st, state['stage_peak'][st])
    for s_, k in sites:
        print(f'   {k:4d}  {s_}')
for e in state['events'][-1:]:
    print('== high-water', e['live'], 'stage', e['stage'])
    for s, k in e['sites']:
        print(f'   {k:4d}  {s}')
