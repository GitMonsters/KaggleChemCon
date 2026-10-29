"""Aggregate sj_tprofile counts: inclusive per function name and exclusive per leaf, for chosen stages."""
import pickle, sys, collections
cnt = pickle.load(open('/home/omen/vqcwork/sj1027/tprof_cnt.pkl', 'rb'))
stages = sys.argv[1].split(',')
for st in stages:
    incl = collections.Counter(); leaf = collections.Counter(); tot = 0
    for (s, chain), k in cnt.items():
        if s != st:
            continue
        tot += k
        names = chain.split(' > ')
        for nm in set(names):
            incl[nm] += k
        leaf[names[-1]] += k
    print('==', st, tot)
    print('  inclusive:')
    for nm, k in incl.most_common(30):
        print(f'    {k:7d} {nm}')
    print('  leaf:')
    for nm, k in leaf.most_common(15):
        print(f'    {k:7d} {nm}')
