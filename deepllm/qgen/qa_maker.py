import time
from collections import defaultdict
from deepllm.interactors import Agent
from deepllm.api import *


def clean_sent(sent):
    sent = sent.strip().replace(' .', '.').replace('..', '')
    sent = sent.replace(' -', '-').replace("'", "_")
    sent = " ".join(sent.split())
    if not sent.endswith('.'): sent = sent + "."
    return sent


def clean_quest(x0, sent, context):
    x = x0.strip()
    # print('!!! CLEANING:', x)

    assert x, ("Empty!!!!", (sent, context))

    assert 'A:' in x or 'Q:' in x, ('MISSING A: or Q:', x)

    if x[0] in ("'", '"'): x = x[1:]
    if x[-1] in ("'", '"'): x = x[0:-1]

    assert x and x[0:3] in ['Q: ', 'A: '], x0
    return x


def to_quests(agent, question, context, k=3):
    agent.set_pattern(None)
    p = f"""
    With the context "{context}" in mind,
    generate {k} different answers to "{question}".
    Prefix each answer with "A:", but do NOT number them as in "A1: or 1. ".
    After returning each answer, suggest a salient follow-up question to your answer, prefixed with "Q:" . 
    """
    prompt = " ".join(p.split())

    answer = agent.ask(prompt)

    # print('PROMPT:',prompt)
    # print('!!! RETURN:\n',answer)
    # print()

    return answer


def quest2quests(agent, quest, context, k=3):
    t1 = time.time()

    quests_ = to_quests(agent, quest, context, k=k)
    quests0 = quests_.replace('\n\n', '\n').split('\n')
    quests = [clean_quest(q, quest, context) for q in quests0]
    # print('LENS:', len(quests0), len(quests))
    if len(quests) % 2 != 0:
        quests = quests[0:-1]  # fix LLM ignoring instruction
    # assert len(quests) % 2 == 0, (len(quests0), len(quests), quest, quests0)

    pairs = []
    for j, x in enumerate(quests):

        p = x[0:3]
        assert p in ['Q: ', 'A: ']
        x = x[3:]
        if j % 2 == 0:
            assert p == "A: ", (p, x)
            a = x  # answers

            q = quests[j + 1]
            p_ = q[0:3]
            q = q[3:]  # quest: next position
            assert p_ == "Q: ", (p_, q)
            pair = (a, q)
            pairs.append(pair)

    t2 = time.time()
    print('TIME:', round(t2 - t1, 4))
    print('COSTS:', round(agent.dollar_cost(), 4))
    return pairs


def one_quest(agent, quest, context, trim_size=3):
    agent.set_initiator(quest)
    res = quest2quests(agent, quest, context, k=1)
    agent.trim_at(trim_size)
    agent.persist()
    a, q = res[0]
    return a, q


def make_agent():
    agent = Agent(name='QA_generator')
    agent.resume()
    return agent


def localize(local):
    if local:
        local_model()
    else:
        key = os.getenv("OPENAI_API_KEY")
        set_openai_api_key(key)
        # smarter_model()
        cheaper_model()


def recursor(initiator, trim_size=3, max_k=2, max_d=5):
    agent = make_agent()
    seen = {initiator}
    rules = dict()

    def generate(quest, d):

        def thread_end(a, q):
            if a in seen or q in seen:
                return []
            elif d > max_d:
                seen.add(a)
                seen.add(q)
                return [[quest, a]]
            else:
                seen.add(a)
                seen.add(q)
                return None

        pairs = quest2quests(agent, quest, initiator, k=max_k)
        agent.trim_at(trim_size)

        rules[quest] = pairs

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
        save_rules(rules)
        #print('RULES:',rules)


def save_rules(rules, fname="rules.pl"):
    def qt(x):
        x=x.replace("'",'_').replace('"','_')
        return f"'{x}'"

    with open(fname, 'w') as f:
        print('% RULES:', len(rules), file=f)
        for h, bs in rules.items():
            print(f"{qt(h)} -->", file=f)
            for i, (a, q) in enumerate(bs):
                line = f"['A:',{qt(a)}]"
                if q:
                    line = line + f",['Q:',{qt(q)}],{qt(q)}"
                print('    ', line, end="", file=f)
                if i < len(bs) - 1:
                    print(";", file=f)
            print(".", file=f)


def show_mems(agent):
    print('SHORT_TERM_MEMORY SIZE:',
          len(agent.short_mem),
          'LONG_TERM_MEMORY SIZE:',
          len(agent.long_mem),
          'COSTS:', round(agent.dollar_cost(), 4))


def test_qa_maker(fresh=0, local=1):
    localize(local)
    agent = make_agent()
    agent.resume()
    # initiator = "Why do some people think that we live in a simulation?"
    initiator = "How do transformers work?"
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


def test_qa_maker1(fresh=0):
    agent = make_agent()
    agent.resume()
    # quest = "Why do some people think that We live in a simulation?"
    quest = "Why would introducing a planning element in the training of an LLM be a big step toward AGI?"
    print('QUEST0:', quest)
    for a, q in quest2quests(agent, quest, quest):
        print('A:', a)
        print('Q:', q)
        print()
    x = one_quest(agent, quest, 'think clearly')
    print('QA:', x)
    agent.persist()
    if fresh: agent.clear()


if __name__ == "__main__":
    test_qa_maker()
