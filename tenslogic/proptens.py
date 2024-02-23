import torch
from deepllm.params import *
from deepllm.questmaker import SymTable


class VectTable:
    def __init__(self, sym_table):
        self.sym_table = sym_table
        n = len(sym_table)
        self.mat = torch.eye(n)

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


top = "true"
bot = "false"


def tp(M, v):
    """
    one step fixpoint operator
    """
    r = M @ v
    return (r >= 1.0).to(torch.float32)


def tp_n(M, v0):
    """
    iterated fixpoint operator
    """
    oldv = v0
    n = M.shape[0]
    for i in range(n):
        tprint('OLDV:', oldv)
        newv = tp(M, oldv)
        tprint('NEWV: ', newv)
        if torch.allclose(newv, oldv):
            return newv
        oldv = newv


def compute_model(prog):
    vs = dict()
    bools = [top, bot]
    for h, bs in prog:
        if h in bools: continue
        vs[h] = True
        for b in bs:
            if b in bools: continue
            vs[b] = True
    vs = list(vs) + bools

    st = SymTable()
    for v in vs:
        st.add(v)
    n = len(st)
    vt = VectTable(st)

    M = torch.zeros(n, n)
    M[:, n - 1] = 1
    M[n - 2, :] = 1

    for h, bs in prog:
        m = len(bs)
        for b in bs:
            ch = vt.at(h)
            cb = vt.at(b)
            M[ch, cb] = 1 / m

    v0 = M[:, n - 2]

    tprint(M)
    tprint(v0)
    tprint('-----')
    T = M @ v0
    tprint(T)
    tprint()

    res = tp_n(M, v0)
    if torch.allclose(torch.tensor(1.0), res[-1]):
        return None
    res = res[0:-2]
    tprint("RES:", res)
    model = vt.decode(res)
    return model


def test_propvecs():
    vs = (p, q, r, s) = "pqrs"

    prog = [
        (p, [q]),
        (p, [r]),
        (q, [r, s]),
        (r, [top]),
        (bot, [q])
    ]

    model = compute_model(prog)
    print('PROG:')
    for h, bs in prog:
        print(h, ':-')
        for b in bs:
            print(f'  {b}')

    print()
    print('MODEL:', model)


def test_json(path='../deepllm/tests/STATE_SMARTER/out/',
              jname='generative_ai_scientific_concept_explorer_2.json'):
    path=path+jname
    assert exists_file(path),path
    clauses = from_json(path)
    prog = []

    for h, bss in clauses.items():
        for bs in bss:
            if bs == [] or bs == ['fail']: continue
            prog.append((h, bs))

    for h, bss in clauses.items():
        if bss == []:
            prog.append((h, [top]))
        for bs in bss:
            if bs == []:
                prog.append((h, [top]))

    for h, bs in prog:
        print(h, ":-")
        for b in bs:
            print('  ', b)
        print()

    model = compute_model(prog)
    print('PROG:')
    for h, bs in prog:
        print(h, ':-')
        for b in bs:
            print(f'  {b}')

    print()
    print('MODEL:')
    for sent in sorted(model):
        print(sent)
    print('MODEL SIZE:',len(model))


if __name__ == "__main__":
    #test_propvecs()
    #test_json(jname='use_of_tactical_nukes_in_ukraine_war_causal_inference_2.json')
    #test_json(jname='benchmark_qa_on_document_colections_scientific_concept_explorer_2.json')
    test_json(
        path='../deepllm/tests/STATE_LOCAL/out/',
        jname='logic_programming_scientific_concept_explorer_3.json')
