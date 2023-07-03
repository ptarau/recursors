from deepllm.tests.test_configurator import test_configurator
from deepllm.tests.test_params import IS_LOCAL_LLM, PARAMS, test_params
from deepllm.tests.test_horn_prover import test_horn_prover
from deepllm.tests.test_embedders import test_embedders
from deepllm.tests.test_interactors import test_interactors
from deepllm.tests.test_recursors import test_recursors
from deepllm.tests.test_refiners import test_refiners


def test_local_runs():
    IS_LOCAL_LLM[0] = True
    print('PARAMS:', PARAMS())
    test_configurator()
    test_horn_prover()
    test_embedders()
    test_interactors()
    test_recursors()
    test_refiners()
    test_params()
    return True


if __name__ == "__main__":
    assert test_local_runs()
