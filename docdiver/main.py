from time import time
from collections import Counter
import networkx as nx
from sentence_store.main import Embedder
from sentify.main import sentify
from deepllm.api import *
from rephrasers import RelationBuilder

SENT_CACHE = './SENT_CACHE/'
SENT_STORE_CACHE = './SENT_STORE_CACHE/'


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
        self.times = Counter()

        self.saved_file_name = as_local_file_name(
            self.doc_type,
            self.doc_name,
            saved_file_name
        )

        sents, t_convert, t_segment = sentify(
            doc_type,
            doc_name,
            store=SENT_CACHE + self.saved_file_name,
            return_timings=True
        )

        self.emb = Embedder(self.saved_file_name)
        self.emb.store(sents)
        self.times['sentify_conversion'] = t_convert
        self.times['sentify_segmentation'] = t_segment

    def get_sents(self):
        res = self.emb.get_sents()
        return res

    def get_knns(self):
        res = self.emb.knns(self.top_k, as_weights=True)
        return res

    def extract_summary(self, best_k=10, center=None):
        knns = self.get_knns()
        t1 = time()
        sents = self.get_sents()
        max_sents = min(80, max(2 * best_k, len(sents) // 2))
        g = nx.DiGraph()

        if center:
            qknns, _ = self.emb.knn_query(center, max_sents)
            ids = set(i for (i, _) in qknns)
        else:
            ids = set(range(len(sents)))

        print('!!! SELECTED SENTS:', len(ids))
        # for i in ids: print('---',i,sents[i])

        for i, ns in enumerate(knns):
            if i in ids:
                for (n, r) in ns:
                    if n in ids:
                        g.add_edge(i, n, weight=r)

        ranked = nx.pagerank(g)
        ranked = sorted(ranked.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[0:best_k]
        best_ids = sorted(i for (i, _) in ranked)

        res = [(i, sents[i]) for i in best_ids]
        t2 = time()
        self.times['extract_summary'] += t2 - t1
        print('SALIENT SENTENCES:', len(res), 'out of:', len(sents))
        return res

    def summarize(self, best_k=8, mark=1, center=None):
        def emphasize(w, important):
            if w.lower() not in important: return w
            return f":green[{w}]"

        id_sents = self.extract_summary(best_k=best_k, center=center)

        if self.trace:
            for x in id_sents:
                print(x)
            print()
        sents = [s for (_, s) in id_sents]

        text = " ".join(sents)
        sm = SummaryMaker(text, sum_size=best_k, kwd_count=8)
        text = sm.run()
        self.costs += sm.dollar_cost()
        self.times['llm_summary_maker_agent'] += sm.agent.processing_time

        if mark:
            source_words = set(w.lower() for s in sents for w in s.split())
            target_words = text.split()
            target_words = [emphasize(w, source_words) for w in target_words]
            text = " ".join(target_words)
            # text = text.replace(' .', '. ').replace(' ?', '? ')
            text = text.replace('Keyphrases:', '\n\nKeyphrases:')

        if text.startswith('Summary'): return text
        return 'Summary: ' + text

    def show_relation_graph(self, best_k, abstractive=False, show=False, center=None):
        if abstractive:  # more testing needed
            text = self.summarize(best_k=best_k, mark=0)
            text = text.replace('Summary:', '').replace('Keyphrases:', '')
            sents = text.split('. ')
            sents = [s.strip() + '.' for s in sents if s and s != '\n']
        else:
            id_sents = self.extract_summary(best_k=best_k, center=center)
            sents = [s for (_, s) in id_sents]

        kind = ['_extr', '_abstr'][int(abstractive)]

        cent = ""
        if center: cent = "_" + center[0:10]

        rel_agent = RelationBuilder(self.saved_file_name + kind + cent + "_rels")
        _jterm, _url, hfile = rel_agent.from_sents(sents, show=show)

        self.times = self.times | rel_agent.times
        self.costs += rel_agent.dollar_cost()

        assert None not in (rel_agent.pname, rel_agent.jname)

        return hfile, rel_agent.pname, rel_agent.jname

    def review(self, best_k=200, center=None):
        id_sents = self.extract_summary(best_k, center=center)

        if self.trace:
            for x in id_sents:
                print(x)
            print()
        sents = [s for (_, s) in id_sents]
        text = " ".join(sents)
        pr = PaperReviewer(text)
        text = pr.run()
        self.costs += pr.dollar_cost()
        self.times['llm_reviewer_agent'] = pr.agent.processing_time
        return 'Review: ' + text

    def retrieve(self, query, top_k=None):
        if top_k is None: top_k = self.top_k

        sents_rs = self.emb.query(query, top_k)

        return [sent for (sent, r) in sents_rs]

    def ask(self, quest, top_k=20):
        t1 = time()
        sents = self.retrieve(quest, top_k=top_k)
        text = " ".join(sents)
        agent = RetrievalRefiner(text, quest, tname=self.saved_file_name)
        answer_plus = agent.run()
        answer, follow_up = answer_plus.split('==>')
        answer = answer.strip()
        follow_up = follow_up.strip().replace('Follow-up question:', '')
        self.costs += agent.dollar_cost()
        t2 = time()
        self.times['llm_query_agent'] += t2 - t1
        return answer, follow_up

    def get_times(self):
        return self.times | self.emb.get_times()

    def dollar_cost(self):
        # self.costs += self.emb.dollar_cost()
        return self.costs


def test_main1(doc='https://arxiv.org/pdf/2306.14077.pdf'):
    # smarter_model()
    cheaper_model()
    # local_model()
    sd = SourceDoc(doc_type='url', doc_name=doc, threshold=0.5, top_k=3)
    sents = sd.summarize(best_k=20)
    print(sents)


def test_main(

    doc='https://arxiv.org/pdf/2306.14077.pdf',
    quest='How is Horn Clause logic used to refine interaction with LLM dialog threads?'
):
    clear_caches()
    remove_dir(SENT_CACHE)

    print("DOC:", doc)
    print('QUEST:', quest)
    smarter_model()
    # cheaper_model()
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
    print("TIMES:")
    for k, v in sd.get_times().items():
        print(k, '=', v)
    sd.show_relation_graph(60, show=True, center='llm')


if __name__ == "__main__":
    test_main()
