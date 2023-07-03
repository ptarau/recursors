from deepllm.params import *


class Temp:
    def __init__(self):
        self.hi = 'hello'


if __name__ == "__main__":
    CF = PARAMS()
    print(CF)
    CF.TRACE = 1
    d = CF(Temp())
    print(d.hi,d.TRACE)
    assert d.TRACE==1
