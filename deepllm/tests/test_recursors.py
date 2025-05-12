from deepllm.prompters import *
from deepllm.recursors import run_explorer, run_symplanner
from deepllm.refiners import SymPlanner


def test_symplanner():
    plan1 = [
        ("vote for a party", ["vote republican"]),
        ("vote for a party", ["vote democrat"]),
    ]

    plan2 = [
        ("buying an ev", ["ev_lower_maintenence"]),
        ("buying an ev", ["ev_cheaper_over_time", "ev_is_fun_to_drive"]),
        ("buying an ev", ["ev_good_for_environment"]),
        ("ev_lower_maintenence", ["ev_no_oil_change"]),
    ]

    run_symplanner(
        explorer=SymPlanner,
        prompter=verifier_prompter,
        goal="buy_an_ev",
        lim=1,
        plan=plan2,
    )


def run_all():

    # run_explorer(prompter=falsifier_prompter, goal='add high tarifs on imports', lim=2)
    # run_explorer(prompter=verifier_prompter, goal='add high tarifs on imports', lim=2)
    # run_explorer(prompter=falsifier_prompter, goal='reaction of the stock market to high tarifs on imports', lim=2)
    # run_explorer(prompter=verifier_prompter, goal='reaction of the stock market to high tarifs on imports', lim=2)
    test_symplanner()
    return
    run_explorer(prompter=sci_prompter, goal="Logic programming", lim=3)
    run_explorer(prompter=sci_prompter, goal="Generative AI", lim=2)
    run_explorer(prompter=recommendation_prompter, goal="Apocalypse now", lim=2)
    run_explorer(
        prompter=recommendation_prompter, goal="1Q84, by Haruki Murakami", lim=2
    )
    run_explorer(prompter=causal_prompter, goal="Expansion of the Universe", lim=2)
    run_explorer(
        prompter=causal_prompter, goal="Use of tactical nukes in Ukraine war", lim=2
    )
    run_explorer(prompter=conseq_prompter, goal="Use of tactical nukes", lim=2)
    run_explorer(
        prompter=sci_prompter, goal="benchmark QA on document colections", lim=2
    )


def demo():
    run_explorer(prompter=task_planning_prompter, goal="Repair a flat tire", lim=1)
    run_explorer(prompter=sci_prompter, goal="Logic Programming", lim=1)
    run_explorer(
        prompter=sci_prompter, goal="Teaching computational thinking with Prolog", lim=2
    )


def test_recursors():
    # assert run_explorer(prompter=task_planning_prompter, goal='Repair a flat tire', lim=1)
    run_all()


if __name__ == "__main__":
    test_recursors()
