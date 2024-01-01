from deepllm.api import *
from deepllm.embedders import Embedder
from deepllm.refiners import load_ground_truth
from inquisitor import QuestExplorer


class TruthJudge(QuestExplorer):
    def __init__(self, initiator=None, prompter=None, truth_file=None, top_k=None, threshold=None, lim=None):
        assert None not in (initiator, prompter, truth_file, top_k, threshold, lim)
        super.__init__(initiator=initiator, prompter=prompter, lim=lim)

        self.top_k = top_k
        self.threshold = threshold
        self.store = Embedder(truth_file)
        self.truth_file = truth_file
        if not exists_file(self.store.cache()):
            sents = load_ground_truth(truth_file=truth_file)
            self.store.store(sents)

        def appraise(self, g, _trace):
            sents_rs = self.store.query(g, self.top_k)
            z = list(zip(*sents_rs))
            r = sum(z[1]) / self.top_k
            if r > self.threshold:
                ok = True
            else:
                ok = False

            tprint(
                f'RATING of "{self.initiator}->{g}" w.r.t truth in "{self.truth_file}.txt" is {round(r, 4)} --> {ok}')
            tprint('AS AVG. OF NEAREST SENTS:')
            for sent, r in sents_rs: tprint(sent, '->', round(r, 4))
            return ok
