from deepllm.api import *
from sentify.main import sentify
from deepllm.embedders import Embedder
from inquisitor import QuestExplorer,quest_prompter


class TruthJudge(QuestExplorer):
    def __init__(self,
                 initiator=None,
                 prompter=None,
                 file_type=None,
                 truth_file=None,
                 top_k=None,
                 threshold=None,
                 lim=None,
                 local=None
                 ):
        assert None not in (initiator, prompter, file_type, truth_file, top_k, threshold, lim, local)
        super().__init__(initiator=initiator, prompter=prompter, lim=lim, local=local)

        self.top_k = top_k
        self.threshold = threshold
        self.store = Embedder(truth_file)
        self.truth_file = truth_file
        if not exists_file(self.store.cache()):
            sents = sentify(file_type, truth_file)
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

def test_oracles():
    tg=TruthJudge(
        initiator="What is SLD resolution?",
        prompter=quest_prompter,
        file_type='wikipage',
        truth_file='Logic Programming',
        top_k=3,
        threshold=0.50,
        lim=2,
        local=0
    )
    tg.run()


if __name__=="__main__":
    test_oracles()
