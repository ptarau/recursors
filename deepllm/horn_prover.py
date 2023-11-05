import time


def qprove(css0, goal=None, early=False):
    """
    Variant of Algorithm 1 in Dowling and Gallier
     Finds a (minimal) model of a propositional Horn Clause program
     Added goal-driven optional goal-focussed execution (if early=True)
     in which case as soon as the goal is proven it returns the model,
     (possibly incomplete).
    """
    props = dict()
    css = []
    gss = []

    # css0 = [(h, bs) for (h, bss) in css0 for bs in bss]

    for c in css0:
        if isinstance(c, tuple):
            h, bs = c
        else:
            h, bs = c, []

        if goal is not None and goal == h:
            gss.append((h, bs))
        else:
            css.append((h, bs))

    if goal is not None and gss == []:
        return None

    css = gss + css

    for h, bs in css:
        props[h] = False
        for b in bs: props[b] = False

    for h, bs in css:
        if bs == []:
            props[h] = True

    """
    propagate True from facts to rules
    while watching for inconsistencies
    """
    change = True
    while change:
        change = False
        for i, c in enumerate(css):
            if c is None: continue
            h, bs = c
            if all(props[b] for b in bs):
                if h == 'false':
                    print('CONTRADICTION:', h, bs)
                    # return None
                    continue
                if not props[h] and all(props[b] for b in bs):
                    css[i] = None
                    props[h] = True

                    if early and h == goal: break

                    change = True

    model = {p for p, v in props.items() if v}
    if goal is not None and goal not in model: return None

    model = list(sorted(model))
    # print('!!! MODEL:', len(model))

    return model


# tester

def horn_formula(n):
    """
    generates all Horn formulas if size n
    """

    for xs in list_partition(n):
        gen = partition_(xs)
        for xss in gen:
            css = [(xs[0], xs[1:]) if xs[1:] else xs[0] for xs in xss]
            for h in range(n):
                yield h, css


# to make this self-contained, for testing with pypy

def partition_(xs):
    if len(xs) == 1:
        yield [xs]
        return

    first = xs[0]
    for smaller in partition_(xs[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


# from partition as list of list, to list of indices
def part2list_(N, pss):
    res = []
    pl = len(pss)
    for i in range(N):
        for j in range(pl):
            if i in pss[j]:
                res.append(j)
    return res


def list_partition(n):
    xs = list(range(n))
    for pss in partition_(xs):
        yield part2list_(n, pss)


def test_horn_prover(n=7):
    print('testing on all Horn formulas of size =',n)
    # for x in list_partition(3): print(x)
    # for x in horn_formula(3): print(x)

    t1 = time.time()

    yes, no = 0, 0
    for x in horn_formula(n):
        if qprove(x[1], x[0]):
            yes += 1
        else:
            no += 1

    t2 = time.time()
    print(f'yes={yes}, no={no} density={yes / (yes + no)}, time={round((t2 - t1), 2)}s')
    if n==7: assert yes==1234073 and no==4149830

def loop_test():
    tree=(0,[(0,[1,2]),1,2])
    g, css = tree
    r = qprove(css, goal=g)
    print('TREE RES:', r)

    loop=(0,[(0,[1,0,2]),1,2,(3,[4,5]),4,5])
    g,css=loop
    r=qprove(css,goal=g)
    print('LOOP RES from 0:',r)

    loop = (3, [(0, [1, 0, 2]), 1, 2, (3, [4, 5]), 4, 5])
    g, css = loop
    r = qprove(css, goal=g)
    print('LOOP RES from 3:', r)


if __name__ == "__main__":
    pass
    loop_test()
    test_horn_prover()
