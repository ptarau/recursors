from collections import Counter
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
        self.actor = Agent(qa_name)
        self.actor.set_pattern(prompter['quest_p'])

    def ask_actor(self, quest, context):
        assert quest, (quest, context)
        qa_pairs = quest2quests(self.actor, quest, context, k=3)
        return qa_pairs

    def persist(self):
        self.actor.persist()

    def resume(self):
        self.actor.resume()

    def costs(self):
        return {'qa_maker': self.actor.dollar_cost()}


class QuestExplorer:
    """
    Simple Prolog-inspired recursive descent steering LLMs to
    expand conjunctively or disjunctively while
    generating answers + follow-up questions.
    Besides returning a stream of answers we
    also generate a Prolog program to be further
    explored with logic programming tools.
    """

    def __init__(self, initiator=None, prompter=None, lim=None):
        assert None not in (initiator, prompter, lim)
        self.initiator = " ".join(initiator.replace('.', ' ').strip().split())
        self.name = self.initiator.lower().strip().replace(' ', '_')
        self.prompter = prompter
        self.pname = prompter['name']
        self.lim = lim
        self.unf = QAUnfolder(self.name, prompter, lim)
        self.OUT = None
        PARAMS()(self)

    def new_pair(self, g, trace):
        """
        invents a set of new (answer, follow-up question) pairs
        given a goal and the trace of
        the past steps leading to g
        """
        context = to_context(trace, self.initiator)
        qa_pairs = self.unf.ask_actor(g, context)
        for a, q in qa_pairs:
            yield a, q

    def costs(self):
        cost_dict = self.unf.costs()
        return cost_dict

    # end overrides

    def solve(self):
        self.resume()
        self.qrings = Counter()
        self.arings = Counter()
        self.opens = Counter()

        def step(g, gs, d):  # gs is the trace so far, g is quest
            if g in gs:
                self.qrings[g] += 1
                return  # would be a loop otherwise
            self.persist()
            if d >= self.lim:
                self.opens[g] += 1
                yield gs
            else:
                for a, q in self.new_pair(g, gs):
                    if a in gs:
                        self.arings[a] += 1
                        yield gs
                    else:
                        trace = a, (g, gs)
                        yield from step(q, trace, d + 1)

        for gs in step(self.initiator, (), 0):
            yield list(reversed(to_list(gs)))

        self.persist()

    def run(self, printer=print):
        rules = defaultdict(list)
        for gs in self.solve():
            if printer:
                d = len(gs)
                for i in range(d):
                    if i % 2 == 1:
                        q = gs[i - 1]
                        a = gs[i]
                        printer('Q:', q)
                        printer()
                        printer('A:', a)
                        printer()
                        if i < d - 2:
                            q_ = gs[i + 1]
                            rules[q].append([q, a, q_])
                        else:
                            assert i == d - 1
                            rules[q].append([q, a])
                printer()
        if printer:
            printer('Q RINGS:')
            for r in self.qrings.items():
                printer(r)
            printer()
            printer('A RINGS:')
            for r in self.arings.items():
                printer(r)
            printer()
            printer('OPENS:')
            for q in self.opens.items():
                printer(q)
            printer()
            printer('COSTS', self.costs())
        pro_name = f"./OUT/{self.pname}_{self.lim}__{self.name.replace('?','')}.pl"
        save_rules(rules,pro_name)
        printer('SAVED TO:',pro_name)

        self.persist()

    def store_results(self):
        # pro_name = f'{self.OUT}{self.name}_{self.pname}_{self.lim}'
        # to_dcg(self.clauses, pro_name)
        for r in self.solve():
            yield 'TRACE', r
        pass

    # -------- begin overrides -------------

    def resume(self):
        self.unf.resume()

    def persist(self):
        self.unf.persist()


def is_quest(x):
    return x.endswith('?')


def save_rules(rules, fname):
    def qt(x):
        x = x.replace("'", '_').replace('"', '_')
        return f"'{x}'"

    atable = SymTable()
    qtable = SymTable()

    def qsym(q):
        return "q" + str(qtable.add(q))

    def asym(a):
        return "a" + str(atable.add(a))

    with open(fname, 'w') as f:
        print(f"go:-q0(Xs,[]),nl,member(X,Xs),write(X),nl,nl,fail.\n",file=f)
        for _h,bss in rules.items():
          for t in bss:
            lt = len(t)
            if lt == 3:
                q, a, q1 = t
                q = qsym(q)
                a = asym(a)
                q1 = qsym(q1)
                print(f"{q}-->{q}_,{a}_,{q1}.", file=f)
            else:
                assert lt == 2, t
                q, a = t
                assert isinstance(a,str),(q,a)
                q = qsym(q)
                a = asym(a)
                print(f"{q}-->{q}_,{a}_.", file=f)

        for na, a in enumerate(atable.nums):
            print(f"a{na}_-->[{qt('A: ' + a)}].", file=f)
        for nq, q in enumerate(qtable.nums):
            print(f"q{nq}_-->[{qt('Q: ' + q)}].", file=f)


def test_inquisitor(prompter=quest_prompter, lim=5):
    localize(1)
    # initiator = "Why do some people think that we live in a simulation?"
    # initiator = "How does finetuning an LLM work?"
    # initiator = "How to teach grammars with Prolog?"

    # initiator = "Where the idea that subject and object are inseparable leads in Heidegger's Zein und Zeit?"
    # initiator="How would you integrate planning elements into the chains of transformer blocks that make up the neural network of an LLM?"
    initiator = "How to prove that NP and P are distinct?"

    print('INITIATOR:', initiator)
    assert None not in (prompter, initiator, lim)
    r = QuestExplorer(initiator=initiator, prompter=prompter, lim=lim)
    r.run()


if __name__ == "__main__":
    test_inquisitor()
