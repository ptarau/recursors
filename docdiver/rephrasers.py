import json
from sentify.main import sentify, text2file
from deepllm.interactors import Agent, PARAMS, to_json
from deepllm.api import local_model, smarter_model, cheaper_model
# from deepllm.embedders import Embedder
from sentence_store.main import Embedder
from deepllm.vis import visualize_rels

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

hypernym_prompter = dict(
    name="np_to_hypernym",
    target='.json',
    prompt="""
    For each of the noun phrases separated by ";" in the folowing text
    extract, when possible, its one or two words key concept. 
    Then, using it, generate triplets connecting the concept with a "kind of" link to
    a salient more general concept or hypernym when available.
    Return your result as JSON list of ("S:","V:","O:") JSON triplets.
    Here is the text:
    
    "$text"
    """
)


class Generalizer(Agent):
    def generalize(self, nouns):

        self.set_pattern(hypernym_prompter['prompt'])
        text = "; ".join(nouns)
        text = self.ask(text=text)
        return text

    def post_process(self, _quest, text):
        lines = text.split('\n')
        keepers = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            keepers.append(line)
        return "\n".join(keepers)


def plain_sent(s):
    if isinstance(s, list):
        s = " ".join(map(str, s))
    else:
        s = str(s)
    # print("PLAIN:",s)
    s = s.strip()
    if not s.endswith("."):
        return False
    return s[0:-1].replace("'", "").replace(',', '').replace(';', '').replace('-', '').replace(' ', '').isalpha()


def knn_edges(emb, k=3, as_weights=True):
    assert isinstance(emb, Embedder), emb
    es = []
    ws = []

    for i, ns in enumerate(emb.knns(k, as_weights=as_weights)):
        for (n, r) in ns:
            ws.append(r)
            e = i, r, n
            es.append(e)
    avg = sum(ws) / len(ws)

    if as_weights:  # larger are better
        max_ = max(ws)
        es = [(s, w, o) for (s, w, o) in es if w > (avg + max_) / 2]
    else:  # smaller are better
        min_ = min(ws)
        es = [(s, w, o) for (s, w, o) in es if w < (avg + min_ / 2)]
    return es


def standardize_word(w):
    arts = ['this', 'that', 'of', 'with', 'in', 'into', 'as', 'a', 'an', 'the']
    for art in arts:
        art = art + ' '
        if w.startswith(art):
            w = w[len(art):]

    return w


def standardize_triplet(x):
    """When the object starts with an preposition like
    "to" or "in" move it to the end of the verb."""
    s, v, o = tuple(t.lower() for t in x)
    (a, sp, b) = str.partition(o, ' ')
    if a in {
        'to', 'from', 'in', 'at', 'by', 'over',
        'under', 'on', 'off', 'away', 'through'
    }:
        v = v + sp + a
        o = b
    if v in {'is', 'be', 'are', 'was', 'were', 'has been', 'have been'}: v = 'is a'

    s, o = standardize_word(s), standardize_word(o)
    return s, v, o


def good_noun_phrase(x0):
    x = x0.replace(' ', '').replace("'", '').replace('-', '').replace('.', '')
    ok = x.isalpha() and x not in {
        'it', 'they', 'he', 'she',
        'someone', 'some', 'all', 'any', 'one'
    }
    if not ok: print('*** NOT A GOOD NOUNPHRASE:', x0)
    return ok


def jterm2svos(jterm):
    if isinstance(jterm[0], list):
        svos = [tuple(x) for x in jterm if len(x) == 3 and x[0] != x[2]][1:]
    else:
        svos = [tuple(x.values()) for x in jterm if len(x) == 3]

    svos = [standardize_triplet(x) for x in svos]

    svos = [(s, v, o) for (s, v, o) in svos if
            good_noun_phrase(s) and good_noun_phrase(o)]

    return svos


def collapse_similars(svos):
    wdict = dict()
    for svo in svos:
        for w in svo:
            wdict[w] = w
    ws = list(wdict)
    for w in ws:
        plural = w + 's'
        if plural in wdict:
            wdict[plural] = w
    svos = sorted(set((wdict[s], wdict[v], wdict[o]) for (s, v, o) in svos))
    return svos


def as_json(jtext):
    # print('!!! LLM ANSWER:', jtext)
    jtext = jtext.replace("```json", '').replace("```", '')
    try:
        jterm = json.loads(jtext)
        assert jterm
        return jterm
    except Exception:
        print('*** json exception on LLM answer')
        return None


class RelationBuilder(Agent):

    def from_source(self, kind, source, hypernyms=True, save=True, show=True, max_sents=80, weights=False):
        sents = sentify(kind, source)
        sents = [s.strip() for s in sents if plain_sent(s)]

        if max_sents:
            sents = sents[0:max_sents]

        return self.from_sents(sents, hypernyms=hypernyms, save=save, show=show, weights=weights)

    def from_sents(self, sents, hypernyms=True, save=True, show=True, weights=False):
        text = "\n".join(sents)
        return self.from_canonical_text(text, hypernyms=hypernyms, save=save, show=show, weights=weights)

    def from_canonical_text(self, text, hypernyms=True, save=True, show=True, weights=False):
        prompter = svos_prompter
        self.spill()
        self.set_pattern(prompter['prompt'])
        CF = PARAMS()
        self.pname = CF.OUT + self.name + "_" + prompter['name']
        self.jname = self.pname + prompter['target']
        self.pname = self.pname + ".pl"

        jtext = self.ask(text=text)
        jterm = as_json(jtext)

        if jterm is None:
            return None, None, None

        if save:
            text2file(text, CF.OUT + self.name + "_sents.txt")

            if jterm is None:
                svos = []
                so_set = set()
            else:

                svos = jterm2svos(jterm)

                so_set = sorted(set(x for (s, _, o) in svos for x in (s, o)))
                assert so_set, svos

                so_embedder = Embedder('so_embedder_' + self.name)
                so_embedder.store(so_set)

                es = knn_edges(so_embedder, k=3, as_weights=False)

                def from_so_set(x):
                    try:
                        return so_set[x]
                    except IndexError:
                        print('bad index:', x, len(so_set))
                        return f'bad index {x}'

                # print('!!! KNN EDGES:', len(so_set), es)

                def adapt_r(r):
                    if weights:
                        r = int(100 * round(r, 2))
                    else:
                        r = 'hints to'
                    return r

                so_knn_links = [(from_so_set(s), adapt_r(r), from_so_set(o)) for (s, r, o) in es]

                if weights:
                    so_knn_links = sorted(so_knn_links, key=lambda x: x[1])

                so_knn_links = [(x, str(r), o) for (x, r, o) in so_knn_links]
                so_knn_links = so_knn_links[0:min(len(so_set), len(svos)) // 3]

                svos.extend(so_knn_links)

            if hypernyms:
                g = Generalizer(self.name)
                hjtext = g.generalize(so_set)
                hjterm = as_json(hjtext)
                if hjterm is not None:
                    # print('HYPERS:\n', json.dumps(jterm))
                    hsvos = jterm2svos(hjterm)
                    hsvos = [(x, 'is a', y) for (x, _, y) in hsvos]
                    svos.extend(hsvos)

            svos = collapse_similars(svos)

            to_json(svos, self.jname)
            to_prolog(svos, self.pname)

            fname = CF.OUT + self.name + "_" + prompter['name']
            url, hfile = visualize_rels(svos, fname=fname, show=show)
            return jterm, url, hfile

        return jterm, None, None


def to_prolog(svos, fname):
    def q(x):
        x = str(x).replace("'", ' ')
        return f"'{x}'"

    with open(fname, 'w') as g:
        svos = sorted(svos, key=lambda x: x[1])
        for s, v, o in svos:
            line = f"rel({q(s)},{q(v)},{q(o)})."
            print(line, file=g)


def test_relationizer():
    # smarter_model() # only GPT4 works? GPT3.5 seems ok!
    # page_name = 'open world assumption'
    # page_name = 'logic_programming'
    page_name = 'enshittification'
    # page_name = "Generative artificial intelligence"
    # page_name = 'Artificial general intelligence'
    agent = RelationBuilder(page_name)
    # _text = agent.from_source('wikipage', page_name)
    _text = agent.from_source('txt', '../data/legal/us_const.txt')

    # print(text)
    # text2file(text, agent.jname)
    print(agent.dollar_cost())


if __name__ == "__main__":
    # local_model()
    # cheaper_model()
    smarter_model()
    test_relationizer()
