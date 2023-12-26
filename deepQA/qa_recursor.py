import time
from collections import defaultdict
from deepllm.interactors import Agent
from deepllm.api import *
from questmaker import *





def recursor(initiator, trim_size=4, max_k=3, max_d=2, local=0):
    localize(local)
    agent = make_agent()

    seen = set()
    rules = dict()
    qtable = SymTable()
    atable = SymTable()

    def getq(q):
        return "q" + str(qtable.add(q))

    def geta(a):
        return "a" + str(atable.add(a))

    getq(initiator)

    def generate(quest, d):

        def thread_end(a, q):
            if a in seen or q in seen:
                print('!!! DEJA VU:', a, q)
                res = []
            elif d > max_d:
                res = [[quest, a]]
            else:
                res = None
            seen.add(a)
            seen.add(q)
            return res

        pairs = quest2quests(agent, quest, initiator, k=max_k)
        agent.trim_at(trim_size)

        npairs = [(geta(a), getq(q)) for (a, q) in pairs if a not in seen]

        rules[getq(quest)] = npairs

        for a, q in pairs:
            end = thread_end(a, q)
            if end is not None:
                yield end
            else:
                for trace in generate(q, d + 1):
                    yield [[quest, a]] + trace

    for trace in generate(initiator, 0):
        if trace:
            show_mems(agent)
            agent.persist()
            yield trace

    rules = trim_rules(rules, atable, qtable)
    ensure_path('OUT/')
    save_rules(initiator, rules, atable, qtable, f'OUT/rules_{local}.pl')


def trim_rules(rules, atable, qtable):
    defined = set()
    referred = set()
    looper = set()
    for h, bs in rules.items():
        defined.add(h)
        for (_, q) in bs:
            if q in defined:
                looper.add(q)
            else:
                referred.add(q)

    trimmed = dict()
    seen = set()
    for h, bs in rules.items():
        cs = []
        for (a, q) in bs:
            if q in looper:
                if a not in seen:
                    seen.add(a)
                    cs.append((a, ()))
            elif q in referred - defined:
                if a not in seen:
                    seen.add(a)
                    cs.append((a, ()))
            else:
                if a not in seen and q not in seen:
                    seen.add(a)
                    cs.append((a, q))

            if cs:
                trimmed[h] = cs
            else:
                trimmed[h] = [([], ())]
    return trimmed


def save_rules(initiator, rules, atable, qtable, fname):
    def qt(x):
        x = x.replace("'", '_').replace('"', '_')
        return f"'{x}'"

    with open(fname, 'w') as f:

        line = f"""go:-
                 start(Xs,[]),nl,nl,
                 member(X,Xs),write(X),nl,fail.
           """
        print(line, file=f)

        print('% RULES:', len(rules) + len(atable) + len(qtable) + 1, file=f)

        q0 = f"q{qtable.syms[initiator]}"
        line = f"\nstart-->{q0}_,{q0}."
        print(line, '\n', file=f)

        for h, bs in rules.items():

            print(f"{h} -->", file=f)
            for i, (a, q) in enumerate(bs):
                line = f"{a}"
                if q:
                    line = line + f",{q + '_'},{q}"
                print('    ', line, end="", file=f)
                if i < len(bs) - 1:
                    print(";", file=f)
            print(".", file=f)
        for na, a in enumerate(atable.nums):
            print(f"a{na}-->[{qt('A: ' + a)}].", file=f)
        for nq, q in enumerate(qtable.nums):
            print(f"q{nq}_-->[{qt('Q: ' + q)}].", file=f)


def show_mems(agent):
    print('SHORT_TERM_MEMORY SIZE:',
          len(agent.short_mem),
          'LONG_TERM_MEMORY SIZE:',
          len(agent.long_mem),
          'COSTS:', round(agent.dollar_cost(), 4))


def test_qa_maker(fresh=0):
    agent = make_agent()

    agent.resume()
    initiator = "Why do some people think that we live in a simulation?"
    # initiator = "How does finetuning an LLM work?"
    initiator = "How to teach grammars with Prolog?"
    # initiator = "Where the idea that subject and object are inseparable leads in Heidegger's Zein und Zeit?"
    # initiator=" How would you integrate planning elements into the chains of transformer blocks that make up the neural network of an LLM?"
    print('INITIATOR:', initiator)
    for thread in recursor(initiator):
        print('\nTHREAD:\n')
        for qa in thread:
            assert len(qa) == 2, ("HERE", qa)
            q, a = qa
            print('Q:', q)
            print('A:', a)
            print()
        print('-----\n')
    if fresh: agent.clear()
    print('SHORT_TERM_MEMORY SIZE:',
          len(agent.short_mem),
          'LONG_TERM_MEMORY SIZE:',
          len(agent.long_mem),
          'COSTS:', round(agent.dollar_cost(), 4))


if __name__ == "__main__":
    test_qa_maker()
