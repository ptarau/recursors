import json
from scipy.spatial.distance import cdist
import numpy as np
import torch
from sentify.main import sentify, text2file
from deepllm.interactors import Agent, PARAMS, to_json
from deepllm.api import local_model, smarter_model, cheaper_model
from deepllm.embedders import Embedder
from deepllm.vis import visualize_rels

witt_prompter_txt = dict(
    name="tractatus_style",
    target=".txt",
    prompt="""
 In the style of Wittgenstein's Tractatus summarize the essential
 atomic facts that he would state about the following text. 
 Return the result as tree representing details following 
 the numerotation scheme of the Tractatus.
 Here is the text to work on: 
 
 "$text".
    """
)

svos_prompter = dict(
    name="fact_to_svos",
    target='.json',
    prompt="""Split each sentence in the the following text into SVO triplets. 
    If the sentence is too complex, split it first into simple sentences. 
    When it is clear form the context, replace pronouns with what they refer to.
    Trim down the subject and object parts to their essential noun phrases.
    Return your result as JSON list of ("S:","V:","O:") JSON triplets.
    Here is the text:     
    "$text"
    """
)


class Factualizer(Agent):
    # def __init__(self, name, **kwargs):
    #    super().__init__(name, **kwargs)

    def factify(self, prompter, kind, source, save=True):
        CF = PARAMS()
        self.set_pattern(prompter['prompt'])
        self.outf = CF.OUT + self.name + "_" + prompter['name'] + prompter['target']
        sents = sentify(kind, source)
        text = "\n".join(sents)
        text = self.ask(text=text)  # text is the
        if save:
            text2file(text, self.outf)
        return text

    def post_process(self, _quest, text):
        lines = text.split('\n')
        keepers = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not line[0].isdigit():
                continue
            keepers.append(line)
        return "\n".join(keepers)


def plain_sent(s):
    s = s.strip()
    if not s.endswith("."):
        return False
    return s[0:-1].replace("'", "").replace(',', '').replace(';', '').replace('-', '').replace(' ', '').isalpha()


def knn_edges(encs, k=2):
    assert k + 1 < len(encs), (k + 1, '<', len(encs))
    cos_scores = torch.from_numpy(cdist(encs, encs, metric='cosine'))
    top_results = torch.topk(cos_scores, largest=False, k=k + 1)
    m = top_results[1]  # indices
    r = top_results[0]  # similarity ranks
    s = m.size()

    es = []
    rs = []
    ws = []
    for i in range(s[0]):
        for j in range(1, s[1] - 1):
            rs.append(r[i, j])
            w = r[i, j]
            ws.append(float(w))
            e = i, w, int(m[i, j]),
            es.append(e)
    avg = sum(ws) / len(ws)
    ws_ = [w for w in ws if w < avg]
    avg_ = sum(ws_) / len(ws_)
    es = [(s, w, o) for (s, w, o) in es if w < avg_]

    return es


def standardize(x):
    x = x.lower()
    a, _b, c = str.partition(x, ' ')
    if a in ['a', 'an', 'the']: x = c
    return x


def good_noun_phrase(x):
    x = x.lower().replace(' ','')
    ok=x.isalpha() and x not in {
        'it', 'they', 'he', 'she',
        'someone', 'some', 'all', 'any', 'one'
    }
    if not ok: print('NO GOOD:',x)
    return ok


def move_prep(x):
    """When the object starts with an preposition like
    "to" or "in" move it to the end of the verb."""
    s,v,o=x
    (a,sp,b)=str.partition(o,' ')
    if a in {
        'to','from','in','at','by','over',
        'under','on','off','away','through'
    }:
        v=v+sp+a
        o=b
    return s,v,o



class RelationBuilder(Agent):

    def run(self, kind, source, so_links=True, save=True, show=True, max_sents=30):
        prompter = svos_prompter
        self.spill()
        self.set_pattern(prompter['prompt'])
        CF = PARAMS()
        self.outf = CF.OUT + self.name + "_" + prompter['name'] + prompter['target']
        sents = sentify(kind, source)
        sents = [s.strip() for s in sents if plain_sent(s)]
        sents=sents[0:max_sents]

        text = "\n".join(sents)
        if save:
            text2file(text,CF.OUT + self.name+"_sents.txt")
        jtext = self.ask(text=text)
        print('LLM ANSWER:', jtext)
        jterm = json.loads(jtext)
        assert jterm
        if save:

            if isinstance(jterm[0], list):
                svos = [tuple(x) for x in jterm if len(x) == 3 and x[0] != x[2]][1:]
            else:
                svos = [tuple(x.values()) for x in jterm if len(x) == 3]

            svos = [(standardize(s), v.lower(), standardize(o)) for (s, v, o) in svos if
                    good_noun_phrase(s) and good_noun_phrase(o)]

            svos = [move_prep(x) for x in svos]

            if so_links:
                so_set = sorted(set(x for (s, _, o) in svos for x in (s, o)))
                so_embedder = Embedder(None)
                so_embeddings = so_embedder.embed(so_set)
                so_knn_links = [(so_set[s], ':', so_set[o]) for (s, r, o) in knn_edges(so_embeddings, k=4)]
                svos.extend(so_knn_links)
            else:
                # ilinks = [(str(i), '', str(i + 1)) for i in range(len(svos) - 1)]
                # ilinks.append((str(len(svos) - 1), '', str(0)))

                embedder = Embedder(None)
                embeddings = embedder.embed(sents)
                knn_links = [(str(s), '', str(o)) for (s, _, o) in knn_edges(embeddings, k=2)]

                slinks = [(str(i), 'S:', s[0]) for (i, s) in enumerate(svos)]
                olinks = [(str(i), 'O:', s[2]) for (i, s) in enumerate(svos)]

                svos.extend(knn_links)
                # svos.extend(ilinks)
                svos.extend(slinks)
                svos.extend(olinks)

            to_json(svos, self.outf)

            fname = CF.OUT + self.name + "_" + prompter['name']
            visualize_rels(svos, fname=fname, show=show)

        return jterm


def test_relationizer():
    #page = 'horn_clause'
    page = 'logic_programming'
    agent = RelationBuilder(page)
    text = agent.run('wikipage', page)
    print(text)
    # text2file(text, agent.outf)
    print(agent.dollar_cost())


def test_rephraser():
    page = 'horn_clause'
    agent = Factualizer(page)
    text = agent.factify(witt_prompter_txt, 'wikipage', page)
    print(text)
    text2file(text, agent.outf)
    print('COST:', agent.dollar_cost())


def test_rephraser1():
    page = 'logic_programming'
    agent = Factualizer(page)
    text = agent.factify(witt_prompter_txt, 'wikipage', page)
    print(text)
    text2file(text, agent.outf)
    print('COST:', agent.dollar_cost())


def test_rephraser2():
    # url = 'https://arxiv.org/pdf/1904.11694.pdf'
    url = 'https://arxiv.org/pdf/1912.10824.pdf'
    agent = Factualizer('url_dif')
    text = agent.factify(witt_prompter_txt, 'url', url)
    print(text)
    text2file(text, agent.outf)
    print(agent.dollar_cost())


def test_rephraser3():
    txt = './data/gpl.txt'
    agent = Factualizer('txt_gpl')
    text = agent.factify(witt_prompter_txt, 'txt', txt)
    print(text)
    text2file(text, agent.outf)
    print(agent.dollar_cost())


if __name__ == "__main__":
    # test_rephraser()

    # local_model()
    cheaper_model()
    #smarter_model()
    test_relationizer()
