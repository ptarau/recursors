from deepllm.api import *

def test_api():
    for x in run_recursor(
        prompter=task_planning_prompter,
        initiator='Build a Chat Bot',
        lim=1
    ):
        jpp(x)
    for x in run_recursor(
        prompter=conseq_prompter,
        initiator='Emergence of superhuman artificial general intelligence',
        lim=1
    ):
        jpp(x)
    for x in run_advisor(
        prompter=conseq_prompter,
        initiator='Practical fusion reactors',
        lim=1
    ): jpp(x)

    for x in run_rater(
        prompter=sci_prompter,
        initiator='Low power circuit design',
        threshold=0.95,
        lim=1
    ):
        jpp(x)

    for x in run_rater(
        prompter=sci_prompter,
        initiator='Low power circuit design',
        threshold=0.50,
        lim=1
    ):
        jpp(x)

    return True


if __name__ == "__main__":
    print(PARAMS())
    assert test_api()
