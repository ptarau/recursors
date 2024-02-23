import networkx as nx
from deepllm.embedders import Embedder
from deepllm.refiners import SummaryMaker
from deepllm.params import jpp
from deepllm.api import *
from sentify.main import sentify


def as_local_file_name(doc_type, doc_name, saved_file_name):
    if not saved_file_name:
        if doc_type == 'url':
            file_name = doc_name.split('/')[-1]
        elif doc_type == 'wikipage':
            file_name = doc_name.replace(' ', '_') + ".txt"
        else:
            file_name = doc_name
    else:
        file_name = saved_file_name

    file_name = file_name.replace('.pdf', '.txt').replace('.PDF', '.txt')
    return file_name


class SourceDoc:
    def __init__(self, doc_type=None, doc_name=None, saved_file_name=None, threshold=None, top_k=None):
        args = (doc_type, doc_name, threshold, top_k)
        assert None not in args, args
        self.doc_type = doc_type
        self.doc_name = doc_name
        self.threshold = threshold
        assert top_k > 0, top_k
        self.top_k = top_k
        self.saved_file_name = as_local_file_name(
            self.doc_type,
            self.doc_name,
            saved_file_name
        )
        sents = sentify(
            doc_type,
            doc_name,
            store='in/' + self.saved_file_name
        )
        self.emb = Embedder('out/' + doc_name)
        self.emb.store(sents)

    def get_sents(self):
        return self.emb.get_sents()

    def get_knns(self):
        return self.emb.knns(self.top_k)

    def extract_summary(self, best_k=10):
        knns = self.get_knns()
        g = nx.DiGraph()
        for i, ns in enumerate(knns):
            for (n, r) in ns:
                g.add_edge(i, n, weight=r)
        ranked = nx.pagerank(g)
        ranked = sorted(ranked.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[0:best_k]
        best_ids = sorted(i for (i, _) in ranked)
        sents = self.get_sents()
        return [(i, sents[i]) for i in best_ids]

    def summarize(self,best_k=8,trace=False):
        id_sents= self.extract_summary(best_k=2*best_k)

        if trace:
            for x in id_sents:
                print(x)
            print()
        sents = [s for (_, s) in id_sents]
        text=" ".join(sents)
        sm=SummaryMaker(text)
        text=sm.run()
        return 'Summary: '+text

    def ask(self, query):
        sents_rs = self.emb.query(query, self.top_k)
        z = list(zip(*sents_rs))
        r = sum(z[1]) / self.top_k

        print('COSTS:', self.emb.dollar_cost())

        return [sent for (sent, r) in sents_rs]

    def heads(self):
        centers = self.emb.cluster()
        # print("!!!!",centers)
        return centers


def test_quest(doc='red.txt', quest='Who concealed his visage?'):
    sd = SourceDoc(doc_type='txt', doc_name=doc, threshold=0.5, top_k=3)
    a = sd.ask(quest)
    print('Q:', quest)
    print('A:', a)
    print('---------\n')
    for x in sd.heads(): jpp(x)


def test_main(doc='https://arxiv.org/pdf/2306.14077.pdf'):
    #smarter_model()
    #cheaper_model()
    local_model()
    sd = SourceDoc(doc_type='url', doc_name=doc, threshold=0.5, top_k=3)
    sents = sd.summarize(best_k=20)
    print(sents)


if __name__ == "__main__":
    test_main()
    # test_quest(doc='summarize.txt', quest='How is text graph built and ranked?')
    # test_main(doc='sein.txt', quest="Where the idea that subject and object are inseparable leads in Heidegger's Zein und Zeit?")
