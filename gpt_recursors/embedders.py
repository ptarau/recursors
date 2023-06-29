import openai
import torch
from params import to_pickle, from_pickle, exists_file, PARAMS


class Embedder:
    def __init__(self, model=None):
        self.total_toks = 0
        self.CACHES = None
        PARAMS()(self)

    def embed(self, sents):
        response = openai.Embedding.create(
            input=sents,
            model=self.model
        )
        embeddings = [response['data'][i]['embedding'] for i in range(len(response['data']))]
        toks = response["usage"]["total_tokens"]
        self.total_toks += toks
        return torch.tensor(embeddings)

    def __call__(self, sents, cache):
        if cache is not None:
            cache = self.CACHES + cache + ".pickle"
        if cache is not None and exists_file(cache):
            x = from_pickle(cache)
            return x
        x = self.embed(sents)
        if cache is not None:
            to_pickle(x, cache)
        return x

    def dollar_cost(self):
        return self.total_toks * 0.0004 / 1000


def cos_sim(embs, qembs):
    return torch.nn.CosineSimilarity(dim=-1, eps=1e-08)(embs,qembs)


def knn_pairs(encs, k=3):
    """
    extracts edges of the directed knn-graph
    associating k closest neighbors to each node
    representing a sentence via its embedding
    """
    cos_scores = cos_sim(encs, encs)
    top_results = torch.topk(cos_scores, k=k + 1)
    m = top_results[1]  # indices
    r = top_results[0]  # similarity ranks
    s = m.size()

    es = []
    for i in range(s[0]):
        for j in range(1, s[1] - 1):
            e = i, int(m[i, j]), r[i, j]
            es.append(e)

    return es


def test_gpt_agents():
    e = Embedder()
    res = e(["The dog barks to the moon", "The cat sits on the mat"], cache=None)
    print(res)
    print('COST:', e.dollar_cost(), len(res[0]))
    print('SIM:', cos_sim(res[0], res[1]))
    print(PARAMS())


if __name__ == "__main__":
    test_gpt_agents()
