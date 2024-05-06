from deepllm import __version__
from deepllm.prompters import *
from deepllm.params import *
from deepllm.recursors import AndOrExplorer, show_clauses, show_model, show_svos,vis_svos
from deepllm.vis import browse
from deepllm.refiners import Advisor, Rater, TruthRater, AbstractMaker, SummaryMaker, PaperReviewer,RetrievalRefiner

def get_version():
    return __version__
def activate_svos():
    GPT_PARAMS['TO_SVOS']=True
    LOCAL_PARAMS['TO_SVOS'] = True

def deactivate_svos():
    GPT_PARAMS['TO_SVOS']=False
    LOCAL_PARAMS['TO_SVOS'] = False


def smarter_model():
    IS_LOCAL_LLM[0] = False
    #GPT_PARAMS['model'] = "gpt-4"
    GPT_PARAMS['model'] ='gpt-4-turbo'
    openai.api_base = GPT_PARAMS['API_BASE']
    GPT_PARAMS['ROOT'] = "./STATE_SMARTER/"
    PARAMS()


def cheaper_model():
    IS_LOCAL_LLM[0] = False
    GPT_PARAMS['model'] = "gpt-3.5-turbo"
    openai.api_base = GPT_PARAMS['API_BASE']
    GPT_PARAMS['ROOT'] = "./STATE/"
    PARAMS()

def local_model():
    IS_LOCAL_LLM[0]=True
    openai.api_base = LOCAL_PARAMS['API_BASE']
    PARAMS()


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
    rater = TruthRater(initiator=initiator, prompter=prompter, truth_file=truth_file, threshold=threshold, lim=lim)
    yield from rater.run()


def run_abstract_maker(topic=None, keywords=None):
    assert None not in (topic, keywords)
    recursor = AbstractMaker(topic=topic, keywords=keywords)
    return recursor.run()
