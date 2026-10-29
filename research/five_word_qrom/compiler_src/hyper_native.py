from pathlib import Path
import sys
from rank_parity import *
def build_hmr(delta,brot,lowbits,ac,terms):
    import kickmix as km
    delta = require_redundant_sign_bit(delta,brot)
    D,J = delta.shape
    n = (D-1).bit_length()
    A,B,O = ac.shape
    H,L = A-1,B-1
    builder = km.CircuitBuilder()
    qubits = list(builder.create_quantum_register(n,'selection'))
    for j in range(J):
        qubits += list(builder.create_quantum_register(brot,f'word_{j}'))
    trace = []
    def gate(name,*ids):
        trace.append((name,*ids))
        getattr(builder,name.lower())(*[qubits[i] for i in ids])
    cache = []
    def cached_basis(size,offset):
        result = {0:None}
        for mask in range(1,size):
            bit = mask.bit_length()-1
            if mask.bit_count()==1:
                result[mask] = offset+bit
            else:
                parent = mask ^ (1 << bit)
                q = len(qubits)
                qubits.extend(builder.alloc_qubits(1))
                result[mask] = q
                triple = (result[parent],offset+bit,q)
                gate('CCX',*triple)
                cache.append(triple)
        return result
    high = cached_basis(A,lowbits)
    low = cached_basis(B,0)
    if len(qubits)>128:
        raise ValueError(f'cache layout needs {len(qubits)} qubits')
    basis = [high[i] for i in range(1,A)]+[low[i] for i in range(1,B)]
    groups = [[n+j*brot+bit] + ([n+j*brot+brot-1] if bit==brot-2 else [])
              for j in range(J) for bit in range(brot-1)]
    def target_bits(direction):
        return [q for j in set_bits(direction) for q in groups[j]]
    def linear_output(control,direction):
        for q in target_bits(direction):
            gate('CX',control,q)
    def product(a,b,direction):
        if not a or not b or not direction:
            return
        if a == b:
            for i in set_bits(a):
                linear_output(basis[i],direction)
            return
        p = (a & -a).bit_length()-1
        first = list(set_bits(a ^ (1 << p)))
        for i in first:
            gate('CX',basis[i],basis[p])
        if b >> p & 1:
            b ^= a ^ (1 << p)
        choices = b & ~(1 << p)
        assert choices
        q = (choices & -choices).bit_length()-1
        second = list(set_bits(b ^ (1 << q)))
        for i in second:
            gate('CX',basis[i],basis[q])
        pivot,*others = target_bits(direction)
        for out in others:
            gate('CX',pivot,out)
        gate('CCX',basis[p],basis[q],pivot)
        for out in others:
            gate('CX',pivot,out)
        for i in reversed(second):
            gate('CX',basis[i],basis[q])
        for i in reversed(first):
            gate('CX',basis[i],basis[p])
    high_only = [sum(int(ac[h,0,j]) << j for j in range(O)) for h in range(A)]
    low_only = [sum(int(ac[0,l,j]) << j for j in range(O)) for l in range(B)]
    low_only[0] = 0
    def correction(destination,left,right,direction):
        for a in set_bits(left):
            for b in set_bits(right):
                index = (a+1)|(b+1)
                if index < len(destination):
                    destination[index] ^= direction
    for term in terms:
        pairs = term['factors']
        direction = term['target_direction']
        for i in range(0,len(pairs),2):
            h1,l1 = pairs[i]
            if i+1 == len(pairs):
                product(h1,l1 << H,direction)
            else:
                h2,l2 = pairs[i+1]
                # (h1+l2)(h2+l1) + h1*h2 + l1*l2
                # equals h1*l1 + h2*l2 over GF(2).
                product(h1 ^ (l2 << H),h2 ^ (l1 << H),direction)
                correction(high_only,h1,h2,direction)
                correction(low_only,l1,l2,direction)
    for q in target_bits(high_only[0]):
        gate('X',q)
    for mask in range(1,A):
        linear_output(high[mask],high_only[mask])
    for mask in range(1,B):
        linear_output(low[mask],low_only[mask])
    for a,b,q in reversed(cache):
        builder.del_and(qubits[a],qubits[b],qubits[q])
    circuit = builder.finish_circuit()
    native = int(circuit.operation_statistics().num_toffoli_like)
    assert native == sum(t[0]=='CCX' for t in trace)
    return circuit,trace,{'num_qubits':int(circuit.num_qubits),
        'toffoli_count':native,'clean_cached_functions':len(cache),
        'cache_compute_toffolis':len(cache), 'cache_HMR_cleanup':len(cache),
        'gates':len(trace)}

