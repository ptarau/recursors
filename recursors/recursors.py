from collections import defaultdict

from params import *
from prompters import *
from interactors import Agent, clean_up, to_list, from_text
from horn_prover import qprove
from tools import in_stack

def ask_for_clean(agent, g, context):
    answer = agent.ask(g=g, context=context)
    agent.spill()

    xs = from_text(answer)
    res = clean_up(xs)
    return res


def just_ask(agent, g, context):
    answer = agent.ask(g=g, context=context)
    agent.spill()
    return answer


class Unfolder:
    """
    return pair of AND/OR agents
    steering LLM interactions
    """

    def __init__(self, name, prompter, lim):
        pname = prompter['name']
        and_name = f'{name}_{pname}_and_{lim}'
        or_name = f'{name}_{pname}_or_{lim}'

        self.and_ = Agent(and_name)
        self.or_ = Agent(or_name)

        self.and_.set_pattern(prompter['and_p'])
        self.or_.set_pattern(prompter['or_p'])

    def ask_and(self, goal, context):
        return ask_for_clean(self.and_, goal, context)

    def ask_or(self, goal, context):
        return ask_for_clean(self.or_, goal, context)

    def persist(self):
        self.and_.persist()
        self.or_.persist()

    def resume(self):
        self.and_.resume()
        self.or_.resume()

    def costs(self):
        return {'and': self.and_.dollar_cost(), 'or': self.or_.dollar_cost()}


class AndOrExplorer:
    """
    Simple Prolog-inspired recursive descent steering LLMs to
    expand conjunctively or disjunctively while
    staying focussed via a controlled history of
    explored goals while unfolding new ones.
    Besides returning a stream of answers we
    also generate a Prolog program to be further
    explored with logic programming tools.
    """

    def __init__(self, initiator=None, prompter=None, lim=1, strict=False):
        assert initiator is not None
        assert prompter is not None
        self.initiator = initiator
        self.name = initiator.lower().strip().replace(' ', '_')
        self.pname = prompter['name']
        self.lim = lim
        self.strict = strict
        self.unf = Unfolder(self.name, prompter, lim)
        self.clauses = defaultdict(list)
        self.facts = dict()
        self.OUT=None
        PARAMS()(self)

    def new_clause(self, g, trace):
        """
        invents a set of new clauses
        given a goal and the trace of
        the past steps leading to g
        """
        or_context = to_context(trace, self.initiator)
        hs = self.unf.ask_or(g, or_context)
        and_context = to_context((g, trace), self.initiator)
        for h in hs:
            if h==g or in_stack(h,trace): continue
            bs = self.unf.ask_and(h, and_context)  # invent their bodies
            if h in bs: continue
            yield h, bs

    # begin overrides
    def resume(self):
        self.unf.resume()

    def persist(self):
        self.unf.persist()

    def appraise(self, g, _trace):
        # return g[0] in "CILKPE" # just to test it trims model
        return True

    def costs(self):
        return self.unf.costs()

    # end overrides

    def solve(self):

        self.resume()

        def step(g, gs, d):
            if g in gs: return
            self.persist()
            if d >= self.lim:
                if self.appraise(g, gs):
                    self.facts[g] = True
                    yield g, gs
            else:
                hs = []
                for h, bs in self.new_clause(g, gs):
                    if not self.appraise(h, (g, gs)): continue
                    hs.append(h)
                    trace = h, (g, gs)
                    self.clauses[g].append([h])
                    self.clauses[h].append(bs)
                    for b in bs:
                        yield from step(b, trace, d + 1)
                if self.strict and len(hs) > 1: self.clauses['false'].append(hs)

        for gs in step(self.initiator, (), 0):
            yield list(reversed(to_list(gs)))

        for fact in self.facts: self.clauses[fact].append([])

        self.persist()

        pro_name = f'{self.OUT}{self.name}_{self.pname}_{self.lim}'

        to_prolog(self.clauses, pro_name)

        css = [(h, bs) for (h, bss) in self.clauses.items() for bs in bss]

        # css=list(self.clauses.items())

        model = qprove(css, goal=self.initiator)

        if model is None:
            print('\nNO MODEL ENTAILING:', self.initiator)
        else:
            print('\nMODEL:', len(model), 'facts', '\n')
            for fact in model: print(fact)


def run_explorer(goal=None, prompter=None, lim=None):
    assert None not in (prompter, goal, lim)
    r = AndOrExplorer(initiator=goal, prompter=prompter, lim=lim)
    seen = dict()
    for a in r.solve():
        print('\nTRACE:')
        # a=sorted(a)
        for x in a: print(x)
        print()
        a = tuple(a)
        if a in seen:
            print('--------------SEEN:', a)
        else:
            seen[a] = True

        if len(a) != len(set(a)):
            print('--------------REPEATED:', a)
    r.unf.persist()
    c1 = r.unf.or_.dollar_cost()
    c2 = r.unf.and_.dollar_cost()
    print('COSTS in $:', {'and': c1, 'or': c2, 'total': c1 + c2})


def to_prolog(clauses, fname, neck=":-"):
    def quote(x):
        return "'" + x + "'"

    suf = '.nat' if neck == ":" else ".pl"

    path = fname + suf
    ensure_path(path)
    with open(path, 'w') as f:
        print('% CLAUSES:', file=f)
        for h, bss in clauses.items():
            for bs in bss:
                if bs == [] or bs == ['fail']: continue
                body = ',\n    '.join(map(quote, bs)) + "."
                print(quote(h), neck + '\n    ' + body, file=f)
        for h, bss in clauses.items():
            if bss == []:
                print(quote(h) + ".", file=f)
            for bs in bss:
                if bs == []:
                    print(quote(h) + ".", file=f)
                elif bs == ['fail']:
                    print(quote(h), neck + ' ' + 'fail.', file=f)


def to_context(trace, topgoal):
    if not trace: return topgoal
    context = ".\n".join(reversed(to_list(trace))) + ".\n"
    # print('!!!! CONTEXT:',context, '!!!!\n')
    return context


def run_all():
    run_explorer(prompter=sci_prompter, goal='Logic programming', lim=2)
    run_explorer(prompter=sci_prompter, goal='Generative AI', lim=2)
    run_explorer(prompter=recommendation_prompter, goal='Apocalypse now', lim=2)
    run_explorer(prompter=recommendation_prompter, goal='1Q84, by Haruki Murakami', lim=2)
    run_explorer(prompter=causal_prompter, goal='Expansion of the Universe', lim=2)
    run_explorer(prompter=causal_prompter, goal='Use of tactical nukes in Ukraine war', lim=2)
    run_explorer(prompter=conseq_prompter, goal='Use of tactical nukes', lim=2)
    run_explorer(prompter=sci_prompter, goal='benchmark QA on document colections', lim=2)


def demo():
    #run_explorer(prompter=goal_prompter, goal='Repair a flat tire', lim=1)
    #run_explorer(prompter=sci_prompter, goal='Logic Programming', lim=1)
    run_explorer(prompter=sci_prompter, goal='Teaching computational thinking with Prolog', lim=2)


if __name__ == "__main__":
    pass
    run_all()
    #demo()
