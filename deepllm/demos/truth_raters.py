# from collections import defaultdict
from sentence_store.main import Embedder
from deepllm.prompters import *
from deepllm.recursors import *


def load_ground_truth(truth_file="logic_programming"):
    with open(f"{PARAMS().DATA}{truth_file}.txt", "r") as f:
        sents = f.read().split("\n")
    return [s for s in sents if s]


class TruthRater(AndOrExplorer):
    """
    recursor enhanced with ability to look-up
    how close a given fact is to the set of
    gound-truth facts
    """

    def __init__(self, truth_file=None, threshold=None, **kwargs):
        assert None not in (truth_file, threshold)
        super().__init__(**kwargs)
        self.threshold = threshold
        self.store = Embedder(truth_file)
        self.truth_file = truth_file
        if not exists_file(self.store.cache(".bin")):
            sents = load_ground_truth(truth_file=truth_file)
            self.store.store(sents)
        self.top_k = PARAMS().TOP_K

    def clear(self):
        self.store.clear()

    def appraise(self, g, _trace):
        sents_rs = self.store.query(g, self.top_k)
        z = list(zip(*sents_rs))
        r = sum(z[1]) / self.top_k
        # sents=map(str,sents_rs)

        # tprint('!!!!!', r, '>', self.threshold)
        if r > self.threshold:
            ok = True
        else:
            ok = False
        tprint(
            f'RATING of "{self.initiator}->{g}" w.r.t truth in "{self.truth_file}.txt" is {round(r, 4)} --> {ok}'
        )
        tprint("AS AVG. OF NEAREST SENTS:")
        for sent, r in sents_rs:
            tprint(sent, "->", round(r, 4))
        return ok
