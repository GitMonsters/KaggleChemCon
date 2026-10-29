# Minimal QROM stub
from dataclasses import dataclass
import numpy as np

@dataclass
class QROM:
    k: int = 0
    selection_registers = ()
    target_registers = ()
    num_controls = 0
    data = None
    
    @classmethod
    def build_from_data(cls, data, target_bitsizes=(1,), target_shapes=((1,),)):
        qrom = cls()
        data_arr = np.asarray(data)
        qrom.data = [data_arr.astype(np.uint64)]
        qrom.target_bitsizes = target_bitsizes
        qrom.target_shapes = target_shapes
        qrom.num_controls = 0
        from qualtran import Register, Side, QDType
        sel_bits = max(1, int(np.ceil(np.log2(max(1, data_arr.shape[0])))))
        # Create simple dtype
        sel_dtype = QDType(sel_bits)
        tgt_dtype = QDType(target_bitsizes[0] if target_bitsizes else 1)
        qrom.selection_registers = [Register('a', dtype=sel_dtype, bitsize=sel_bits, side=Side.THRU)]
        qrom.target_registers = [Register('z', dtype=tgt_dtype, shape=target_shapes[0] if target_shapes else (), bitsize=target_bitsizes[0] if target_bitsizes else 1, side=Side.THRU)]
        return qrom
