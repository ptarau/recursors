from .prompters import *
from .params import *
from .recursors import AndOrExplorer
from .refiners import Advisor, Rater, TruthRater, AbstractMaker

def run_recursor(initiator=None, prompter=None, lim=None):
    assert None not in (prompter, initiator, lim)
    recursor = AndOrExplorer(initiator=initiator, prompter=prompter, lim=lim)
    yield from recursor.run()

def run_advisor(initiator=None, prompter=None, lim=None):
    assert None not in (prompter, initiator, lim)
    recursor = Advisor(initiator=initiator, prompter=prompter, lim=lim)
    yield from recursor.run()

def run_rater(initiator=None, prompter=None, lim=None, threshold=None):
    assert None not in (prompter, initiator, lim, threshold)
    recursor = Rater(initiator=initiator, prompter=prompter, lim=lim, threshold=threshold)
    yield from recursor.run()

def run_truth_rater(initiator=None, prompter=None, truth_file=None, threshold=None, lim=None):
    assert None not in (initiator, prompter, truth_file, threshold, lim)
    r = TruthRater(initiator=initiator, prompter=prompter, truth_file=truth_file, threshold=threshold, lim=lim)
    yield from super().run()

def run_abstract_maker(topic=None,keywords=None):
    assert None not in (prompter, initiator, lim)
    recursor = AbstractMaker(topic=topic,keywords=keywords)
    return recursor.run()
