import numpy as np


class SymTable:
    def __init__(self):
        self.syms = dict()
        self.nums = []

    def add(self, sym):
        n = self.syms.get(sym)
        if n is None:
            n = len(self.nums)
            self.syms[sym] = len(self.nums)
            self.nums.append(sym)
        return n

    def __contains__(self, sym):
        return sym in self.syms

    def __len__(self):
        return len(self.nums)

    def __repr__(self):
        return str(self.syms)


class VectTable:
    def __init__(self, sym_table):
        self.sym_table = sym_table
        n = len(sym_table)
        self.mat = np.eye(n)

    def encode(self, sent):
        """sentence to vector"""
        i = self.sym_table.syms[sent]
        return self.mat[i]

    def at(self, sent):
        """sentence to index"""
        i = self.sym_table.syms[sent]
        return i

    def decode(self, vect):
        """ vector to list of sentences """
        sents = []
        for i, x in enumerate(vect):
            x = int(x)
            if x:
                s = self.sym_table.nums[i]
                sents.append(s)
        return sents


def digest(prog, vs):
    st = SymTable()

    for v in vs:
        st.add(v)
    st.add('true')
    st.add('false')

    n = len(st)

    vt = VectTable(st)

    M = np.zeros((n, n))

    for j in range(n):
        M[n - 2, j] = 1
    for i in range(n):
        M[i, n - 1] = 1

    for h, bs in prog:
        m = len(bs)
        for b in bs:
            ch = vt.at(h)
            cb = vt.at(b)
            M[ch, cb] = 1 / m

    v0 = M[:, n - 2].T

    # p\q is read as q->p, head name vertical

    print(M)
    print()
    print(v0)
    print(v0.shape)

    T = M @ v0
    print(T)

    print()

    v1 = np.array([0, 1, 1, 0, 1, 0]).T
    print(v1)
    T1 = vmul(M, v1)
    print('t1:', T1)
    print()

    res = tp(M, v0)
    if np.allclose(1, res[-1]):
        return None
    res = res[0:-2]
    print("RES:")
    print(res)
    model = vt.decode(res)
    print('MODEL:', model)


def vmul(M, v):
    r = M @ v
    print('@@@@:', r)
    r=[r >= 1]
    return r[0]


def tp(M, v0):
    oldv = v0
    n = M.shape[0]
    for i in range(n):
        print('OLDV:', oldv)
        newv = vmul(M, oldv)
        print('NEWV: ', newv)
        if np.allclose(newv, oldv):
            return newv
        oldv = newv


vs = (p, q, r, s) = "pqrs"

prog = [
    (p, [q]),
    (p, [r]),
    (q, [r, s]),
    (r, ['true']),
    ('false', [q])
]


def test_propvecs():
    digest(prog, vs)


if __name__ == "__main__":
    test_propvecs()
