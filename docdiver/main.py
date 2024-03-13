from time import time
import networkx as nx
from deepllm.embedders import Embedder
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
    def __init__(self, doc_type=None, doc_name=None, saved_file_name=None, threshold=None, top_k=None, trace=False):
        args = (doc_type, doc_name, threshold, top_k)
        assert None not in args, args
        self.doc_type = doc_type
        self.doc_name = doc_name
        self.threshold = threshold
        assert top_k > 0, top_k
        self.top_k = top_k
        self.trace = trace
        self.costs = 0
        self.time = 0
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
        t1 = time()
        knns = self.get_knns()
        t2 = time()
        g = nx.DiGraph()
        for i, ns in enumerate(knns):
            for (n, r) in ns:
                g.add_edge(i, n, weight=r)
        ranked = nx.pagerank(g)
        ranked = sorted(ranked.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[0:best_k]
        best_ids = sorted(i for (i, _) in ranked)
        sents = self.get_sents()
        t3 = time()
        print('TIME knns:', round(t2 - t1, 2), 'ranking:', round(t3 - t2, 2))
        self.time = round(t3 - t1, 2)
        return [(i, sents[i]) for i in best_ids]

    def summarize(self, best_k=8, mark=1):
        def emphasize(w, important):
            if w.lower() not in important: return w
            return f":green[{w}]"

        id_sents = self.extract_summary(best_k=best_k)
        print('RAW_SENTS:', len(id_sents))

        if self.trace:
            for x in id_sents:
                print(x)
            print()
        sents = [s for (_, s) in id_sents]

        text = " ".join(sents)
        sm = SummaryMaker(text, sum_size=best_k, kwd_count=8)
        text = sm.run()
        self.costs += sm.dollar_cost()
        self.time += sm.agent.processing_time
        source_words = set(w.lower() for s in sents for w in s.split())
        target_words = text.split()
        if mark:
            target_words = [emphasize(w, source_words) for w in target_words]
            text = " ".join(target_words)
            #text = text.replace(' .', '. ').replace(' ?', '? ')
            text = text.replace('Keyphrases:', '\n\nKeyphrases:')
        if text.startswith('Summary'): return text
        return 'Summary: ' + text

    def review(self, best_k=200):
        id_sents = self.extract_summary(best_k)
        print('RAW_SENTS:', len(id_sents))

        if self.trace:
            for x in id_sents:
                print(x)
            print()
        sents = [s for (_, s) in id_sents]
        text = " ".join(sents)
        pr = PaperReviewer(text)
        text = pr.run()
        self.costs += pr.dollar_cost()
        self.time += pr.agent.processing_time
        return 'Review: ' + text

    def retrieve(self, query, top_k=None):
        if top_k is None: top_k = self.top_k

        sents_rs = self.emb.query(query, top_k)
        print('EMBEDDING COSTS:', self.emb.dollar_cost())
        return [sent for (sent, r) in sents_rs]

    def ask(self, quest, top_k=20):
        t1 = time()
        sents = self.retrieve(quest, top_k=top_k)
        text = " ".join(sents)
        agent = Retrievalrefiner(text, quest, tname=self.saved_file_name)
        answer_plus = agent.run()
        answer, follow_up = answer_plus.split('==>')
        answer = answer.strip()
        follow_up = follow_up.strip().replace('Follow-up question:', '')
        self.costs += agent.dollar_cost()
        t2 = time()
        self.time = round(t2 - t1, 2)
        return answer, follow_up

    def heads(self):
        centers = self.emb.cluster()
        # print("!!!!",centers)
        return centers

    def dollar_cost(self):
        self.costs += self.emb.dollar_cost()
        return self.costs


def test_main1(doc='https://arxiv.org/pdf/2306.14077.pdf'):
    # smarter_model()
    # cheaper_model()
    local_model()
    sd = SourceDoc(doc_type='url', doc_name=doc, threshold=0.5, top_k=3)
    sents = sd.summarize(best_k=20)
    print(sents)


def test_main(
    doc='https://arxiv.org/pdf/2306.14077.pdf',
    quest='How is Horn Clause logic used to refine interaction with LLM dialog threads?'
):
    print("DOC:", doc)
    print('QUEST:', quest)
    # smarter_model()
    cheaper_model()
    # local_model()
    sd = SourceDoc(doc_type='url', doc_name=doc, threshold=0.5, top_k=3)
    sents = sd.retrieve(quest, top_k=20)
    print('RELEVANT SENTENCES:\n', len(sents))
    for s in sents:
        print(s)
    print('-' * 20, '\n')
    answer, follow_up = sd.ask(quest, top_k=20)
    print('ANSWER:\n', answer)
    print()
    print('FOLLOW_UP QUESTION:\n', follow_up)
    print()
    print("COSTS: $", round(sd.dollar_cost(), 4))


if __name__ == "__main__":
    test_main()
