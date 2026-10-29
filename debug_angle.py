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
ak = angle_ks[0]
qrom = ak.qrom
if qrom.num_controls != 0 or len(qrom.data) != 1:
    raise ValueError("bad1")
data = np.asarray(qrom.data[0])
if data.ndim != 2 or min(data.shape) == 0:
    raise ValueError("bad2")
registers = {register.name: register for register in ak.signature}
selection_bits = int(registers["a"].bitsize)
brot_bits = int(registers["z"].bitsize)
_, batch_size = data.shape
if brot_bits < 1 or np.any(data < 0) or np.any(data >= 1 << brot_bits):
    raise ValueError("QROM data must contain unsigned brot-bit words")
print('ok')
