from deepllm.tests.test_refiners import *
from sentify.wikifetch import run_wikifetch, CF


def test_truth_rater(goal=None, prompter=None, truth_file=None, threshold=None, lim=None):
    assert None not in (goal, prompter, truth_file, threshold, lim)
    r = TruthRater(initiator=goal, prompter=prompter, truth_file=truth_file, threshold=threshold, lim=lim)
    r.unf.resume()
    for a in r.solve():
        print('\nTRACE:')
        for x in a:
            print(x)
        print()
    r.unf.persist()
    c = r.costs()
    print('COSTS in $:', c)


def run_all():
    CF.DATA = PARAMS().DATA
    run_wikifetch()

    test_truth_rater(prompter=sci_prompter, goal='Unification algorithm', truth_file='logic_programming',
                     threshold=0.10, lim=1)
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

    test_advisor(prompter=causal_prompter, goal='Biased AI', lim=1)

    test_abstract_maker1()
    test_abstract_maker2()


if __name__ == "__main__":
    run_all()
