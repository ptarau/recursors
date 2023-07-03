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

    c = r.costs()
    print('COSTS in $:', c)


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


def run_all():
    run_wikifetch()
    test_abstract_maker1()
    test_abstract_maker2()
    test_advisor(prompter=causal_prompter, goal='Biased AI', lim=1)
    test_truth_rater(prompter=sci_prompter, goal='Teaching computational thinking with Prolog',
                     truth_file='computational_thinking', threshold=0.50, lim=2)
    test_truth_rater(prompter=sci_prompter, goal='Artificial general intelligence',
                     truth_file='artificial_general_intelligence', threshold=0.50, lim=2)

    test_rater(prompter=causal_prompter, goal='The Fermi paradox', threshold=0.60, lim=2)

    test_rater(prompter=conseq_prompter, goal='P = NP', threshold=0.50, lim=3)

    test_advisor(prompter=recommendation_prompter, goal='The Godfather', lim=2)
    test_rater(prompter=recommendation_prompter, goal='The Godfather', threshold=0.20, lim=2)

    test_rater(prompter=sci_prompter, goal='Logic programming', threshold=0.5, lim=3)
    test_advisor(prompter=recommendation_prompter, goal='The Godfather', lim=2)

    test_advisor(prompter=conseq_prompter, goal='Disproof the Riemann hypothesis', lim=2)
    test_advisor(prompter=conseq_prompter, goal='Proof the Riemann hypothesis', lim=2)

    test_rater(prompter=sci_prompter, goal='Low power circuit design', threshold=0.50, lim=2)

    test_advisor(prompter=recommendation_prompter, goal='Interstellar', lim=1)


def test_refiners():
    assert test_abstract_maker1()
    # assert test_abstract_maker2()
    assert test_rater(prompter=sci_prompter, goal='Low power circuit design', threshold=0.50, lim=1)
    assert test_rater(prompter=causal_prompter, goal='The Fermi paradox', threshold=0.50, lim=1)
    # test_truth_rater(prompter=sci_prompter, goal='Unification algorithm',truth_file='logic_programming', threshold=0.10, lim=1)


if __name__ == "__main__":
    test_refiners()
