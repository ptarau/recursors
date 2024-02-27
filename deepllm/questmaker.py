from deepllm.interactors import Agent
from deepllm.api import *


def localize(local):
    if local:
        local_model()
    else:
        key = API_KEY[0]
        set_openai_api_key(key)
        # smarter_model()
        cheaper_model()


def make_agent(name='QA_generator'):
    agent = Agent(name=name)
    agent.resume()

    return agent


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


def is_quest(x):
    return x.endswith('?')


def clean_sent(sent):
    sent = sent.strip().replace(' .', '.').replace('..', '')
    sent = sent.replace(' -', '-').replace("'", "_")
    sent = " ".join(sent.split())
    if not sent.endswith('.'): sent = sent + "."
    return sent


def clean_quest(x0, sent, context):
    x = x0.strip()
    for i in range(1,6):
        x=x.replace(f'A{i}','A').replace(f'Q{i}','Q')
        if x.startswith(f'{i}.'): x=x[3:]
    #print('!!! CLEANING:', x)

    assert x, ("Empty!!!!", (sent, context))

    if not ('A:' in x or 'Q:' in x):
        print('!!!! MISSING A: or Q:', x)
        return None

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
    quests_ = to_quests(agent, quest, context, k=k)

    # print('!!!!!! QUESTS FROM LLM:',quests_,'END QUESTS\n\n')

    quests0 = quests_.replace('\n\n', '\n').split('\n')

    quests = [clean_quest(q, quest, context) for q in quests0]

    #print('!!!!!! CLEANED FROM LLM:', len(quests))

    if None in quests:
        print('*** None in quests', quests)
        return []  # TODO clean up
    if len(quests) % 2 == 1:
        print('*** Odd number of quests')
        # for q in quests[-2:]:
        #    print('!!!',q)
        #bad = quests[-1]
        # assert not is_quest(bad),bad
        quests = quests[0:-1]
        # return []

    pairs = []
    for j, x in enumerate(quests):
        m = j % 2

        p = x[0:3]
        if m == 0:
            # even A:
            # assert p in ['A: '],[quest,j,x,len(quests)]
            if p != 'A: ':
                return pairs[0:-1]
        else:
            # odd! Q:
            # assert p in ['Q: '],[quest,j,x,len(quests)]
            if p != 'Q: ':
                return pairs
        x = x[3:]
        if j % 2 == 0:
            a = x  # answers
            q = quests[j + 1]  # quest: next position
            #p_ = q[0:3]
            q = q[3:]
            pair = (a, q)
            pairs.append(pair)

    return pairs


def one_quest(agent, quest, context, trim_size=3):
    agent.set_initiator(quest)
    res = quest2quests(agent, quest, context, k=1)
    agent.trim_at(trim_size)
    agent.persist()
    a, q = res[0]
    return a, q


def test_questmaker():
    print('TESTING:')
    localize(0)
    agent = make_agent()
    # agent.resume()
    quest = "What is a neural network?"
    qs = quest2quests(agent, quest, "", k=3)
    print('QUEST:', quest)
    for a, q in qs:
        print('A:', a)
        print('Q:', q)
        print()
    # agent.persist()


if __name__ == "__main__":
    test_questmaker()
