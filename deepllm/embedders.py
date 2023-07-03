from scipy.spatial.distance import cdist
import numpy as np
import openai
from .params import to_pickle, from_pickle, PARAMS


class Embedder:
    """
    embeds a set of sentences using an LLM
    and store them into a vector store
    """

    def __init__(self, cache_name):
        self.total_toks = 0
        self.cache_name = cache_name
        self.CACHES = None
        self.emebedding_model = None
        self.LOCAL_LLM = None
        PARAMS()(self)

    def cache(self):
        return self.CACHES + self.cache_name + ".pickle"

    def embed(self, sents):
        response = openai.Embedding.create(
            input=sents,
            model=self.emebedding_model
        )
        embeddings = [response['data'][i]['embedding'] for i in range(len(response['data']))]
        toks = response["usage"]["total_tokens"]
        self.total_toks += toks
        return embeddings

    def store(self, sents):
        """
        embeds and caches the sentences
        """
        f = self.cache()
        embeddings = self.embed(sents)
        to_pickle((sents, embeddings), f)

    def query(self, query_sent, top_k):
        """
        fetches the store
        """
        sents, embeddings = from_pickle(self.cache())
        query_embeddings = self.embed([query_sent])
        dm = cdist(embeddings, query_embeddings, metric='cosine')
        dm = [1 - d[0] for d in dm]  # cosinus similarity vector
        ids = list(np.argpartition(dm, -top_k)[-top_k:])
        rids = [(i, dm[i]) for i in ids]
        rids.sort(reverse=True, key=lambda x: x[1])
        print('!!!', rids)
        answers = [(sents[i], dm[i]) for (i, _) in rids]
        return answers

    def __call__(self, quest, top_k):
        return self.query(quest, top_k)

    def dollar_cost(self):
        if self.LOCAL_LLM: return 0.0
        return self.total_toks * 0.0004 / 1000
