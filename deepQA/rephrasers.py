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
    prompt="""Split each sentence in the the following text
    into an SVO triplets. If the sentence is too complex, split
    it first into sumple sentences. 
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


def knn_edges(encs, k=1):
    assert k + 1 < len(encs), (k + 1, '<', len(encs))
    cos_scores = torch.from_numpy(cdist(encs, encs, metric='cosine'))
    top_results = torch.topk(cos_scores, k=k + 1)
    m = top_results[1]  # indices
    r = top_results[0]  # similarity ranks
    s = m.size()

    es = []
    rs = []
    for i in range(s[0]):
        for j in range(1, s[1] - 1):
            rs.append(r[i, j])
            e = i, r[i, j], int(m[i, j]),
            es.append(e)
    return es


class RelationBuilder(Agent):
    def relationize(self, kind, source, save=True, show=True):
        prompter = svos_prompter
        self.spill()
        self.set_pattern(prompter['prompt'])
        CF = PARAMS()
        self.outf = CF.OUT + self.name + "_" + prompter['name'] + prompter['target']
        sents = sentify(kind, source)
        sents = [s.strip() for s in sents if plain_sent(s)]

        text = "\n".join(sents)
        jtext = self.ask(text=text)
        print(jtext)
        if save:
            jterm = json.loads(jtext)
            to_json(jterm, self.outf)
            svos = [tuple(x.values()) for x in jterm]

            ilinks = [(str(i), '', str(i + 1)) for i in range(len(svos) - 1)]
            ilinks.append((str(len(svos)-1),'',str(0)))

            embedder = Embedder(None)
            embeddings = embedder.embed(sents)
            knn_links = [(str(s), '', str(o)) for (s, _, o) in knn_edges(embeddings, k=2)]

            slinks = [(str(i), 'S:', s[0]) for (i, s) in enumerate(svos)]
            olinks = [(str(i), 'O:', s[2]) for (i, s) in enumerate(svos)]

            svos.extend(ilinks)
            svos.extend(knn_links)
            svos.extend(slinks)
            svos.extend(olinks)

            fname = CF.OUT + self.name + "_" + prompter['name']
            visualize_rels(svos, fname=fname, show=show)

        return jterm


def test_relationizer():
    page = 'horn_clause'
    agent = RelationBuilder(page)
    text = agent.relationize('wikipage', page)
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
    # cheaper_model()
    smarter_model()
    test_relationizer()
