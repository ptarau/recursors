from collections import defaultdict
from deepllm.recursors import *


def parse_plan(text_plan):
    from natlog.parser import parse

    def fix(cs):
        h, bs = cs
        h = h[0]
        bs = [b[0] for b in bs]
        return h, bs

    css = [fix(cs) for (cs, r) in parse(text_plan, rule=True)]

    return css


def to_paths(css):
    clauses = defaultdict(list)
    for cs in css:
        # print("! cs=", cs)
        h, bs = cs
        clauses[h].append(bs)

    def build(h):
        bss = clauses[h]
        if not bss:
            yield (h, ())
        else:
            for bs in bss:
                for b in bs:
                    for ps in build(b):
                        yield (h, ps)

    def to_context(ps):
        ls = []
        while ps != ():
            p, qs = ps
            ls.append(p)
            ps = qs
        leaf = ls[-1]
        ls = list((ls[0:-1]))
        ls = ". ".join(ls) + "."
        return leaf, ls

    def trim_last(ps):

        qs = ()
        last = ()
        while ps:
            g, ps = ps
            if ps:
                qs = g, qs
            last = g

        return last, qs

    h0 = css[0][0]
    pss = list(build(h0))
    # lss = [to_context(ps) for ps in pss]
    lss = [trim_last(ps) for ps in pss]
    return clauses, lss


def run_symplanner(initiator=None, prompter=None, lim=None):
    assert None not in (prompter, initiator, lim)
    sp = SymPlanner(initiator=initiator, prompter=prompter, lim=lim)
    yield from sp.run()


class SymPlanner(AndOrExplorer):
    def __init__(self, initiator=None, prompter=None, lim=1, strict=False):
        if isinstance(initiator, tuple):
            initiator = "using a Python plan"
            plan = initiator
        elif isinstance(initiator, str):
            if ":" in initiator and "." in initiator and "'" in initiator:

                plan = parse_plan(initiator)
                h, bs = plan[0]
                initiator = h
                # print("#### PLAN:", plan)
            else:
                #  initator unchanged, not a plan

                plan = None

        # print("@@@@ initiator", initiator)

        super().__init__(initiator=initiator, prompter=prompter, lim=lim, strict=False)

        self.set_human_plan(plan)

    def set_human_plan(self, css):
        if not css:
            self.paths = []
        else:
            self.clauses, self.paths = to_paths(css)

    def proceed(self):
        if not self.paths:
            # for gs in self.step(self.initiator, (), 0):
            #    yield list(reversed(to_list(gs)))
            yield from super().proceed()
        else:
            for leaf, ps in self.paths:
                self.initiator = leaf

                for gs in self.step(leaf, ps, 0):
                    yield list(reversed(to_list(gs)))
