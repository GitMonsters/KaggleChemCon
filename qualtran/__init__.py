# Minimal qualtran stub
from dataclasses import dataclass
from typing import Any

class Side:
    THRU = 'thru'
    LEFT = 'left'
    RIGHT = 'right'

class QDType:
    def __init__(self, num_qubits=None):
        self.num_qubits = num_qubits
    def __eq__(self, other):
        if isinstance(other, QDType):
            return self.num_qubits == other.num_qubits
        return NotImplemented
    def __hash__(self):
        return hash(self.num_qubits)

@dataclass
class Register:
    name: str
    dtype: Any = None
    shape: tuple = ()
    side: str = Side.THRU
    bitsize: int = 1
    def __init__(self, name, dtype=None, shape=(), side=Side.THRU, bitsize=None, **kwargs):
        self.name = name
        self.dtype = dtype
        self.shape = shape if shape is not None else ()
        self.side = side
        if bitsize is not None:
            self.bitsize = bitsize
        elif dtype is not None and hasattr(dtype, 'num_qubits'):
            self.bitsize = dtype.num_qubits
        elif dtype is not None and isinstance(dtype, int):
            self.bitsize = dtype
        else:
            self.bitsize = kwargs.get('bitsize', 1)

@dataclass
class Signature:
    registers: list
    def __init__(self, registers):
        self.registers = list(registers)
    def __iter__(self):
        return iter(self.registers)
    def __eq__(self, other):
        if not isinstance(other, Signature):
            return NotImplemented
        if len(self.registers) != len(other.registers):
            return False
        for r1, r2 in zip(self.registers, other.registers):
            if getattr(r1, 'name', None) != getattr(r2, 'name', None):
                return False
            if getattr(r1, 'bitsize', None) != getattr(r2, 'bitsize', None):
                return False
            if getattr(r1, 'shape', ()) != getattr(r2, 'shape', ()):
                return False
            if getattr(r1, 'dtype') != getattr(r2, 'dtype'):
                return False
        return True
    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

class Bloq:
    pass

class BloqBuilder:
    pass

SoquetT = Any

__all__ = ['Bloq', 'BloqBuilder', 'Register', 'Side', 'Signature', 'SoquetT', 'QDType']
