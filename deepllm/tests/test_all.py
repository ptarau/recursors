from deepllm.api import *
from test_configurator import test_configurator
from test_params import test_params
from test_horn_prover import test_horn_prover
from test_embedders import test_embedders
from test_interactors import test_interactors
from test_recursors import test_recursors
from test_refiners import test_refiners


def run_all():
    test_configurator()
    test_params()
    test_horn_prover()
    test_embedders()
    test_interactors()
    test_recursors()
    test_refiners()


if __name__ == "__main__":
    #smarter_model()
    cheaper_model()
    #local_model()
    run_all()
