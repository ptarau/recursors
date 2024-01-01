from collections import Counter
from deepllm.tools import file2string
from deepllm.recursors import *
from deepllm.questmaker import *

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

    def __init__(self, initiator=None, prompter=None, lim=None, local=None):
        assert None not in (initiator, prompter, lim, local)
        if local:
            localize(local)
        self.local = local
        PARAMS()(self)
        self.initiator = " ".join(initiator.replace('.', ' ').strip().split())
        self.name = self.initiator.lower().strip().replace(' ', '_')
        self.prompter = prompter
        self.pname = prompter['name']
        self.lim = lim
        self.unf = QAUnfolder(self.name, prompter, lim)
        self.OUT = None
        self.pro_name = None

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
        self.rejects = Counter()

        def step(g, gs, d):  # gs is the trace so far, g is quest
            if g in gs:
                self.qrings[g] += 1
                return  # would be a loop otherwise
            self.persist()
            if d >= self.lim:
                self.opens[g] += 1
                yield gs
            elif not self.apprise(g,gs):
                self.rejects[g] += 1
                yield gs
            else:
                for a, q in self.new_pair(g, gs):
                    if q == g or q in gs:
                        self.qrings[q] += 1
                        yield gs
                    elif a in gs:
                        self.arings[a] += 1
                        yield gs

                    else:
                        trace = a, (g, gs)
                        yield from step(q, trace, d + 1)

        for gs in step(self.initiator, (), 0):
            yield list(reversed(to_list(gs)))

        self.persist()

    def run(self, printer=print):
        rules = defaultdict(dict)
        for gs in self.solve():
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
                        rules[q][(q, a, q_)] = True
                    else:
                        assert i == d - 1
                        rules[q][(q, a)] = True
            printer()

        if self.qrings:
            printer('Questions that would have started loops:')
            for r in self.qrings.items():
                printer(r)
            printer()
        if self.arings:
            printer('Answers that would repeat proviously seen answers:')
            for r in self.arings.items():
                printer(r)
            printer()
        if self.opens:
            printer('Questions left open at depth limit, with counts:')
            for q in self.opens.items():
                printer(q)
            printer()

        printer('COSTS', self.costs())
        name = self.name.replace('?', '').lower()
        self.pro_name = f"./OUT/{self.pname}_{self.lim}_{name}_{int(self.local)}.pl"
        save_rules(rules, self.qrings, self.arings, self.opens, self.pro_name)

        printer('Definite Clause Grammar saved to:', [os.path.basename(self.pro_name)])
        jname = f'./OUT/rules_{int(self.local)}.json'
        jrules = [(k, v) for (k, vs) in rules.items() for v in vs]
        printer('Json version saved to:', [os.path.basename(jname)])
        to_json(jrules, jname)

        self.persist()

    def show_dcg(self):
        if self.pro_name is None: return None
        return file2string(self.pro_name)

    # -------- begin overrides -------------

    def apprise(self, _g, _trace):
        return True

    def resume(self):
        self.unf.resume()

    def persist(self):
        self.unf.persist()


def save_rules(rules, qrings, arings, opens, fname):
    def qt(x):
        x = x.replace("'", '_').replace('"', '_')
        return f"'{x}'"

    atable = SymTable()
    qtable = SymTable()

    def qsym(q):
        return "q" + str(qtable.add(q))

    def asym(a):
        return "a" + str(atable.add(a))

    ensure_path(fname)
    with open(fname, 'w') as f:
        print(f"go:-q0(Xs,[]),nl,member(X,Xs),write(X),nl,nl,fail.\n", file=f)
        print('\n% DCG GENERATIVE GRAMMAR RULES:\n', file=f)
        for _h, bss in rules.items():
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
                    assert isinstance(a, str), (q, a)
                    q = qsym(q)
                    a = asym(a)
                    print(f"{q}-->{q}_,{a}_.", file=f)

        print('\n% QUESTION TERMINALS:\n', file=f)
        for nq, q in enumerate(qtable.nums):
            print(f"q{nq}_-->[{qt('Q: ' + q)}].", file=f)

        print('\n% ANSWER TERMINALS:\n', file=f)
        for na, a in enumerate(atable.nums):
            print(f"a{na}_-->[{qt('A: ' + a)}].", file=f)

        print('\n% OPEN QUESTIONS:\n', file=f)
        for x, c in opens.most_common():
            print(f"opens({qt(x)},{c}).", file=f)

        print('\n% LOOP TRIGGERING QUESTIONS:\n', file=f)
        for x, c in qrings.most_common():
            print(f"qrings({qt(x)},{c}).", file=f)

        print('\n% REPEATED ANSWERS:\n', file=f)
        for x, c in arings.most_common():
            print(f"arings({qt(x)},{c}).", file=f)


def test_inquisitor(prompter=quest_prompter, lim=5, local=1):
    # initiator = "Why do some people think that we live in a simulation?"
    # initiator = "How does finetuning an LLM work?"
    # initiator = "How to teach grammars with Prolog?"

    # initiator = "Where the idea that subject and object are inseparable leads in Heidegger's Zein und Zeit?"
    # initiator="How would you integrate planning elements into the chains of transformer blocks that make up the neural network of an LLM?"
    # initiator = "How to prove that NP and P are distinct?"
    # initiator = "How to repair a flat tire?"
    initiator = "How to recognize quickly that someone is talking bs?"

    print('INITIATOR:', initiator)
    assert None not in (prompter, initiator, lim)
    r = QuestExplorer(initiator=initiator, prompter=prompter, lim=lim, local=local)
    r.run()  # printer=lambda *x:None)


if __name__ == "__main__":
    test_inquisitor()
