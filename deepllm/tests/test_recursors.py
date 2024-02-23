from deepllm.prompters import *
from deepllm.recursors import run_explorer


def run_all():
    run_explorer(prompter=sci_prompter, goal='Logic programming', lim=3)
    return
    run_explorer(prompter=sci_prompter, goal='Generative AI', lim=2)
    run_explorer(prompter=recommendation_prompter, goal='Apocalypse now', lim=2)
    run_explorer(prompter=recommendation_prompter, goal='1Q84, by Haruki Murakami', lim=2)
    run_explorer(prompter=causal_prompter, goal='Expansion of the Universe', lim=2)
    run_explorer(prompter=causal_prompter, goal='Use of tactical nukes in Ukraine war', lim=2)
    run_explorer(prompter=conseq_prompter, goal='Use of tactical nukes', lim=2)
    run_explorer(prompter=sci_prompter, goal='benchmark QA on document colections', lim=2)


def demo():
    run_explorer(prompter=task_planning_prompter, goal='Repair a flat tire', lim=1)
    run_explorer(prompter=sci_prompter, goal='Logic Programming', lim=1)
    run_explorer(prompter=sci_prompter, goal='Teaching computational thinking with Prolog', lim=2)


def test_recursors():
    #assert run_explorer(prompter=task_planning_prompter, goal='Repair a flat tire', lim=1)
    run_all()

if __name__ == "__main__":
    test_recursors()
