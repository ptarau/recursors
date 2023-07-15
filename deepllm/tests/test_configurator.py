from deepllm.configurator import *


def test_configurator():
    d = Mdict(**dict(a=1, b=2, c=3, d=4, e=5))
    md = Mdict(a=22, c=33)
    md.b = 0
    print('prompters:', d)
    print('md:', md)
    print('prompters:', md(d))
    CFG = Mdict(TRACE=0)
    print(CFG, type(CFG))

    assert d.a == 22 and d.b == 0 and d.c == 33


if __name__ == "__main__":
    test_configurator()
