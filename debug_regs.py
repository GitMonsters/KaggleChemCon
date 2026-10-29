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
sig0 = angle_ks[0].signature
s1 = angle_ks[1].signature
r0a = sig0.registers[0]; r1a = s1.registers[0]
print(r0a.bitsize, r1a.bitsize, r0a.name, r1a.name)
r0z = sig0.registers[1]; r1z = s1.registers[1]
print(r0z.bitsize, r1z.bitsize, r0z.shape, r1z.shape)
print(r0a.dtype == r1a.dtype)
print(r0z.dtype == r1z.dtype)
