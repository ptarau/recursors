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
        self.agent = Agent('abstractor')
        self.agent.set_pattern(prompter['writer_p'])
        PARAMS()(self)

    def run(self):
        return ask_for_clean(self.agent, g=self.topic, context=self.keywords)


class Advisor(AndOrExplorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pname = decision_prompter['name']
        oname = f'{self.name}_{pname}'
        self.oracle = Agent(name=oname)
        self.oracle.set_pattern(decision_prompter['decider_p'])

    def appraise(self, g, _trace, topgoal):
        # xs = to_list((g, trace))
        # context = ", ".join(xs)

        advice = just_ask(self.oracle, g=g, context=topgoal)

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


def test_advisor(name=None, prompter=None, goal=None, lim=None):
    assert None not in (name, goal, prompter, lim)
    r = Advisor(name=name, prompter=prompter, lim=lim)

    for a in r.solve(goal):
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
    t, a = writer.run()
    print()
    print(t)
    print()
    print(a)
    print()
    print(f'Keywords: {writer.keywords}')
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
    t, a = writer.run()
    print()
    print(t)
    print()
    print(a)
    print()
    print(f'Keywords: {writer.keywords}')
    writer.agent.persist()
    print('Cost: $', writer.agent.dollar_cost())


def demo():
    test_advisor(name='rec_adv', prompter=recommendation_prompter, goal='The Godfather', lim=2)
    return
    test_advisor(name='biased_why_adv', prompter=causal_prompter, goal='Biased AI', lim=1)

    test_advisor(name='r_hyp', prompter=conseq_prompter, goal='Disproof the Riemann hypothesis', lim=2)
    test_advisor(name='r_hyp', prompter=conseq_prompter, goal='Proof the Riemann hypothesis', lim=2)


if __name__ == "__main__":
    pass
    # test_it()
    demo()
