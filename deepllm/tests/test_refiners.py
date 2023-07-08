from deepllm.refiners import *


# from deepllm.demos.wikifetch import run_wikifetch


def test_rater(prompter=None, goal=None, threshold=None, lim=None, ):
    assert None not in (goal, prompter, threshold, lim)
    r = Rater(initiator=goal, prompter=prompter, threshold=threshold, lim=lim)

    for a in r.solve():
        print('\nTRACE:')
        for x in a:
            print(x)
        print()
    print('MODEL:',len(r.logic_model))
    c = r.costs()
    print('COSTS in $:', c)
    return True


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
    return True


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
    return True


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
    return True

def test_refiners():
    assert test_abstract_maker1()
    # assert test_abstract_maker2()
    assert test_advisor(prompter=sci_prompter, goal='Low power circuit design',lim=1)
    assert test_rater(prompter=causal_prompter, goal='The Fermi paradox', threshold=0.60, lim=1)


if __name__ == "__main__":
    test_refiners()
