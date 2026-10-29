from dfthc.build_angle_k import build_angle_ks
from starter.task2_angles.encode import encode_inputs,TREES
from starter.task1_factorization.solution_io import BASELINES_DIRECTORY,read_factorization
from verifier.reference import load_systems
from verifier.score import load_inputs
import numpy as np
systems=load_systems()
path=BASELINES_DIRECTORY/systems['Fe2S2-20o30e']['baseline']
params,_=read_factorization(path)
inputs=load_inputs('Fe2S2-20o30e',params)
tree_edges=np.asarray(TREES['log_depth'](inputs.num_orbitals),dtype=np.int64)
words=encode_inputs(inputs,tree_edges,16,'nearest')
angle_ks=build_angle_ks(words.astype(np.int64),brot=16,batch_size=5)
sig0=angle_ks[0].signature
for i,ak in enumerate(angle_ks):
    regs=list(ak.signature)
    print(i, regs[0].bitsize, regs[1].bitsize, type(regs[0].dtype), type(regs[1].dtype))
