from deepllm.recursors import run_explorer
from deepllm.prompters import verifier_prompter, falsifier_prompter

from natlog import Natlog, natprogs

def run_natlog(natprog="symplan.nat"):
    n = Natlog(file_name=natprog,
               with_lib=natprogs() + "lib.nat", callables=globals())
    next(n.solve('initialize.'))
    next(n.solve('decide_on_EVs.'))
    next(n.solve('decide_on_tariffs.'))

def verify(goal,lim):
    run_explorer(prompter=verifier_prompter, goal=goal, lim=lim)

def falsify(goal,lim):
    run_explorer(prompter=falsifier_prompter, goal=goal, lim=lim)

if __name__=="__main__":
    run_natlog()
