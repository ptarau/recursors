from deepllm.interactors import *

def test_interactors(fresh=1):
    CF = PARAMS()
    name = 'tester'
    DI = CF(Agent(name))
    if not fresh:
        DI.resume()
    else:
        DI.clear()
    DI.pattern = "Explain to a teenager what $thing is in $count sentences."
    a = DI.ask(thing='molecule', count='2-3')
    print(a)
    print('$', DI.dollar_cost())
    #print(DI.__dict__)
    DI.persist()


if __name__ == "__main__":
    assert os.getenv("OPENAI_API_KEY")

    test_interactors()

