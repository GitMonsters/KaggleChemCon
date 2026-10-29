"""How often is the synthesis border value x0 (head word 1 after confluent extraction) zero on verifier-like shots?
Also counts the i=1 prefix zero chart. Usage: python border_freq.py <shots> <workers>"""
import sys, os
from multiprocessing import Pool
SRC = '/home/omen/vqcwork/sj1027/session/source/blocks/dqi/team-fused/source'
sys.path[:0] = [SRC, SRC + '/exp/confluent_extract', '/home/omen/vqcwork/comp/kickmix', '/home/omen/vqcwork/comp/verifiers/dqi']
import numpy as np


def work(args):
    seed, n = args
    import kickmix as km
    import verify_submission_dqi as V
    from model import initial, extract
    spec = V.ProblemSpecDQI.benchmark_instance()
    tables = V.FieldTables(km.GF2Field(spec.degree))
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    rng = np.random.default_rng(seed)
    syn, mbuc, ph = V.sample_shots(spec, tables, rng, n, wp)
    zero = 0; zero_words = np.zeros(70, int)
    for s in syn:
        bank = extract(initial([int(x) for x in s]))
        zero += bank[1] == 0
        zero_words += np.array([w == 0 for w in bank])
    return zero, zero_words


if __name__ == '__main__':
    shots, workers = int(sys.argv[1]), int(sys.argv[2])
    per = shots // workers
    with Pool(workers) as p:
        res = p.map(work, [(1000 + i, per) for i in range(workers)])
    z = sum(r[0] for r in res); zw = sum(r[1] for r in res)
    print('shots', per * workers, 'x0==0:', z, 'zero counts per word (first 10):', zw[:10].tolist())
