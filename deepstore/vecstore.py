import math
import numpy as np
from hnswlib import Index

class VecStore(Index):
    def __init__(self, fname, space='cosine',dim=512):  # 512 for sbert
        super().__init__(space=space, dim=dim)
        self.fname = fname
        self.init()

    def load(self):
        self.load_index(self.fname)

    def save(self):
        self.save_index(self.fname)

    def init(self, N=1024):
        self.init_index(max_elements=N,
                        ef_construction=100,
                        M=64,
                        allow_replace_deleted=True
                        )
        self.set_ef(20)
        self.set_num_threads(8)

    def add(self, xss):
        assert xss.shape[1] == self.dim
        N = xss.shape[0]
        N += self.element_count
        if N > self.max_elements:
            N = max(2 * self.max_elements, 2 ** math.ceil(math.log2(N)))
            self.resize_index(N)
        self.add_items(xss)

    def delete(self,xs):
        pass


    def query(self, qss, k=3):
        assert isinstance(k, int)
        DI = self.knn_query(qss, k, filter=None)
        return DI


def test_vecstore():
    vs = VecStore('temp.bin', dim=3)
    xss = [[11, 22, 33], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    qss = [[7, 8, 9]]
    xss = np.array(xss)
    print(xss)
    print()
    qss = np.array(qss)
    print(qss)

    vs.add(xss)
    vs.save()
    vs_ = VecStore('temp.bin', dim=3)
    vs_.load()
    r = vs_.query(qss)
    print()
    print(r[0])
    print(r[1])


if __name__ == "__main__":
    test_vecstore()
