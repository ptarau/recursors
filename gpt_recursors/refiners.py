from recursors import *
from prompters import *


class AbstractMaker:
    def __init__(self, topic=None, keywords=None):
        assert None not in (topic, keywords)
        self.topic = topic
        self.keywords = keywords
        prompter = sci_abstract_maker
        pname = prompter['name']
        tname = topic.replace(' ', '_').lower()
        self.agent = Agent(f'{tname}_{pname}')
        self.agent.set_pattern(prompter['writer_p'])
        PARAMS()(self)

    def run(self):
        return ask_for_clean(self.agent, g=self.topic, context=self.keywords)


class Rater(AndOrExplorer):
    def __init__(self, threshold=None, **kwargs):
        super().__init__(**kwargs)
        pname = ratings_prompter['name']
        oname = f'{self.name}_{pname}'
        self.oracle = Agent(oname)
        self.oracle.set_pattern(ratings_prompter['rater_p'])
        self.threshold = threshold

    def appraise(self, g, _trace):

        advice = ask_for_clean(self.oracle, g=g, context=self.initiator)
        print(f'\n-----EXPLANATION: {advice}\n---\n')
        if not advice:
            print('*** NO ADVICE FOR:', g)
            return False

        advice = advice[0].split('|')[0].strip()
        if ' ' in advice: advice = advice.split()[1]
        try:
            f = float(advice)
        except Exception:
            print('*** UNPARSED RATING:', advice)
            f = 5
        f = f / 100.0

        ok = f >= self.threshold

        print(f'RATING of "{g}" w.r.t "{self.initiator}" is {round(f, 4)} --> {ok}')

        return ok

    def resume(self):
        super().resume()
        self.oracle.resume()

    def persist(self):
        super().persist()
        self.oracle.persist()

    def costs(self):
        d = super().costs()
        d['oracle'] = self.oracle.dollar_cost()
        return d


class Advisor(AndOrExplorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pname = decision_prompter['name']
        oname = f'{self.name}_{pname}'
        self.oracle = Agent(name=oname)
        self.oracle.set_pattern(decision_prompter['decider_p'])

    def appraise(self, g, _trace):
        # xs = to_list((g, trace))
        # context = ", ".join(xs)

        advice = just_ask(self.oracle, g=g, context=self.initiator)

        print('!!! ADVICE for:', g, advice)

        return 'True.' == advice

    def resume(self):
        super().resume()
        self.oracle.resume()

    def persist(self):
        super().persist()
        self.oracle.persist()

    def costs(self):
        d = super().costs()
        d['oracle'] = self.oracle.dollar_cost()
        return d


def test_advisor(prompter=None, goal=None, lim=None):
    assert None not in (goal, prompter, lim)
    r = Advisor(initiator=goal, prompter=prompter, lim=lim)

    for a in r.solve():
        print('\nTRACE:')
        for x in a:
            print(x)
        print()

    c = r.costs()
    print('COSTS in $:', c)


def test_rater(prompter=None, goal=None, threshold=None, lim=None, ):
    assert None not in (goal, prompter, threshold, lim)
    r = Rater(initiator=goal, prompter=prompter, threshold=threshold, lim=lim)

    for a in r.solve():
        print('\nTRACE:')
        for x in a:
            print(x)
        print()

    c = r.costs()
    print('COSTS in $:', c)


def test_abstract_maker1():
    writer = AbstractMaker(
        topic='logic programming',
        keywords="; ".join([
            'Knowledge representation',
            'Knowledge representation formalism',
            'Knowledge-based systems',
            'Ontology engineering',
            'Conceptual graphs'])
    )
    writer.agent.resume()
    ta = writer.run()
    print()
    print(ta)
    print()
    print(f'\nKeywords: {writer.keywords}')
    writer.agent.persist()
    print('Cost: $', writer.agent.dollar_cost())


def test_abstract_maker2():
    writer = AbstractMaker(
        topic='Large language models',
        keywords="; ".join([
            'Natural language processing',
            'Language modeling',
            'Dependency parsing',
            'Named entity recognition',
            'Sentiment analysis',
            'Word embeddings'
        ])
    )
    writer.agent.resume()
    ta = writer.run()
    print()
    print(ta)
    print()

    print(f'Keywords: {writer.keywords}')
    writer.agent.persist()
    print('Cost: $', writer.agent.dollar_cost())


def demo():
    test_rater(prompter=causal_prompter, goal='the Fermi paradox', threshold=0.60, lim=2)
    return
    test_rater(prompter=conseq_prompter, goal='P = NP', threshold=0.10, lim=3)

    test_advisor(prompter=recommendation_prompter, goal='The Godfather', lim=2)
    test_rater(prompter=recommendation_prompter, goal='The Godfather', threshold=0.20, lim=2)

    test_rater(prompter=sci_prompter, goal='Logic programming', threshold=0.5, lim=3)
    test_advisor(prompter=recommendation_prompter, goal='The Godfather', lim=2)

    test_advisor(prompter=causal_prompter, goal='Biased AI', lim=1)
    test_advisor(prompter=conseq_prompter, goal='Disproof the Riemann hypothesis', lim=2)
    test_advisor(prompter=conseq_prompter, goal='Proof the Riemann hypothesis', lim=2)



if __name__ == "__main__":
    pass
    test_abstract_maker1()
    test_abstract_maker2()
    demo()
