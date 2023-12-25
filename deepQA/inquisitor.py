from deepllm.recursors import *
from questmaker import *

quest_prompter = dict(
    name='quest_prompter',
    quest_p="""
    With the context "$context" in mind,
    generate $k different answers to "$question".
    Prefix each answer with "A:", but do NOT number them as in "A1: or 1. ".
    After returning each answer, suggest a salient follow-up question to your answer, prefixed with "Q:" .
    """
)


class QAUnfolder:
    """
    return pair of AND/OR agents
    steering LLM interactions
    """

    def __init__(self, name, prompter, lim):
        pname = prompter['name']
        qa_name = f'{name}_{pname}_and_{lim}'

        self.act = Agent(qa_name)

        self.act.set_pattern(prompter['quest_p'])

    def ask_act(self, quest, context):
        # return ask_for_clean(self.act, goal, context)
        assert quest,(quest,context)
        qa_pairs = quest2quests(self.act, quest, context, k=3)
        return qa_pairs

    def persist(self):
        self.act.persist()

    def resume(self):
        self.act.resume()

    def costs(self):
        return {'qa_maker': self.act.dollar_cost()}


class QuestExplorer:
    """
    Simple Prolog-inspired recursive descent steering LLMs to
    expand conjunctively or disjunctively while
    generating answers + follow-up questions.
    Besides returning a stream of answers we
    also generate a Prolog program to be further
    explored with logic programming tools.
    """

    def __init__(self, initiator=None, prompter=None, lim=1, strict=False):
        assert initiator is not None
        assert prompter is not None
        self.initiator = " ".join(initiator.replace('.', ' ').strip().split())
        self.name = self.initiator.lower().strip().replace(' ', '_')
        self.prompter = prompter
        self.pname = prompter['name']
        self.lim = lim
        self.strict = strict
        self.unf = QAUnfolder(self.name, prompter, lim)
        self.clauses = defaultdict(list)
        self.facts = dict()
        self.svo = None
        self.OUT = None
        PARAMS()(self)

    def new_clause(self, g, trace):
        """
        invents a set of new clauses
        given a goal and the trace of
        the past steps leading to g
        """
        context = to_context(trace, self.initiator)

        qa_pairs = self.unf.ask_act(g, context)

        for a, q in qa_pairs:
            yield a, (q, ())

        """
        hs = self.unf.ask_or(g, context)
        and_context = to_context((g, trace), self.initiator)
        for h in hs:
            if h == g or in_stack(h, trace): continue
            bs = self.unf.ask_and(h, and_context)  # invent their bodies
            if h in bs: continue
            yield h, bs
        """

    def costs(self):
        cost_dict = self.unf.costs()
        return cost_dict

    # end overrides

    def solve(self):

        self.resume()

        def step(g, gs, d):  # gs is the trace so far
            if g in gs: return  # would be a loop
            self.persist()
            if d >= self.lim:
                if self.appraise(g, gs):
                    self.facts[g] = True
                    yield g, gs
            elif not g:
                yield g
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

        self.persist()

        for fact in self.facts: self.clauses[fact].append([])
        css = [(h, bs) for (h, bss) in self.clauses.items() for bs in bss]

        self.trim_clauses()
        self.save_results()

    def save_results(self):
        pro_name = f'{self.OUT}{self.name}_{self.pname}_{self.lim}'

        # to_prolog(self.clauses, pro_name)
        print('!!!! CLAUSES:')
        for x in self.clauses:
            if not x: continue
            print(x)
            print()
            print('-' * 20)

    def run(self):
        yield 'PROMPTER', self.prompter
        for r in self.solve():
            yield 'TRACE', r
        yield 'CLAUSES', dict(self.clauses)
        yield 'MODEL', self.logic_model

        yield 'COSTS', self.costs()

    # -------- begin overrides -------------

    def resume(self):
        self.unf.resume()

    def persist(self):
        self.unf.persist()

    def appraise(self, g, _trace):
        """
        to be overriden by refiners
        """
        return True

    def trim_clauses(self):
        """
        can be overriden : for now, only atoms in the model
        will be kept
        """
        pass


def test_qa_maker(prompter=quest_prompter, lim=2):
    initiator = "Why do some people think that we live in a simulation?"
    # initiator = "How does finetuning an LLM work?"
    initiator = "How to teach grammars with Prolog?"

    # initiator = "Where the idea that subject and object are inseparable leads in Heidegger's Zein und Zeit?"
    # initiator=" How would you integrate planning elements into the chains of transformer blocks that make up the neural network of an LLM?"
    initiator = "How to prove that NP and P are distinct?"

    print('INITIATOR:', initiator)
    assert None not in (prompter, initiator, lim)
    r = QuestExplorer(initiator=initiator, prompter=prompter, lim=lim)
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


if __name__ == "__main__":
    test_qa_maker()
