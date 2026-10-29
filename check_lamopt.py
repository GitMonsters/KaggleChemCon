import numpy as np
for name in ['bp_opt_fin.npz', 'bp_opt_g.npz', 'bp_opt_lam.npz', 'bp_opt_lamfin.npz', 'bp_opt_final.npz']:
    try:
        data = np.load('lamopt_output/runs/' + name, allow_pickle=False)
        print(name + ': keys=' + str(list(data.keys())))
        if 'brot' in data:
            print('  brot=' + str(data['brot']))
    except Exception as e:
        print(name + ': ' + str(e))