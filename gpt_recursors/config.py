class Mdict:
    """
    meta-dict gnerator

    wraps a dict object d to support d.x=...
    notation instead of d['x']= ...
    it also gives back the usual dict view
    and supports sending its content to
    an arbitrary object, as attributes
    """

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return str(self.__dict__)

    def as_dict(self):
        """
        dict view
        """
        return self.__dict__

    def __call__(self, other):
        """
        transfers its attributes as attributes of other
        """
        other.__dict__.update(self.as_dict())
        return other


def test_config():
    d = Mdict(**dict(a=1, b=2, c=3, d=4, e=5))
    md = Mdict(a=22, c=33)
    md.b = 0
    print('d:', d)
    print('md:', md)
    print('d:', md(d))
    CFG = Mdict(TRACE=0)
    print(CFG, type(CFG))


if __name__ == "__main__":
    test_config()
