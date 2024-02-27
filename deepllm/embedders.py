import os
from time import time
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
import openai
from deepllm.params import to_pickle, from_pickle, PARAMS, ensure_openai_api_key, GPT_PARAMS
from sentence_transformers import SentenceTransformer


# SBERT API

def sbert_embed(sents):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sents)
    return embeddings


# LLM API

def llm_embed_old(emebedding_model, sents):
    response = openai.Embedding.create(
        input=sents,
        model=emebedding_model
    )
    embeddings = [response['data'][i]['embedding'] for i in range(len(response['data']))]
    toks = response["usage"]["total_tokens"]
    return embeddings, toks


def llm_embed_new(emebedding_model, sents):
    CF = PARAMS()

    client = openai.OpenAI(
        api_key=ensure_openai_api_key(),
        base_url=GPT_PARAMS['API_BASE']
    )

    response = client.embeddings.create(
        input=sents,
        model=emebedding_model
    )

    embeddings = [response.data[i].embedding for i in range(len(response.data))]
    toks = response.usage.total_tokens
    return embeddings, toks


def get_llm_embed_method():
    try:
        if int(openai.__version__[0]) > 0:
            return llm_embed_new
    except Exception:
        pass
    return llm_embed_old


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
        loc=""
        if self.LOCAL_LLM:
            loc="_local"

        return self.CACHES + self.cache_name + loc + ".pickle"

    def embed(self, sents):
        t1 = time()
        if self.LOCAL_LLM:
            embeddings = sbert_embed(sents)
        else:
            llm_embed = get_llm_embed_method()
            embeddings, toks = llm_embed(self.emebedding_model, sents)
            self.total_toks += toks
        t2 = time()
        print('TIME embed:', round(t2 - t1, 2))
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
        # print('!!!', rids)
        answers = [(sents[i], dm[i]) for (i, _) in rids]
        return answers

    def knns(self, top_k):
        assert top_k > 0, top_k
        t1 = time()
        top_k += 1  # as diagonal is excluded
        sents, embeddings = from_pickle(self.cache())
        dm = cdist(embeddings, embeddings, metric='cosine')
        t2 = time()
        ns = []
        for i in range(len(sents)):
            dm_i = [1 - d[i] for d in dm]
            ids = list(np.argpartition(dm_i, -top_k)[-top_k:])
            rids = [(j, dm_i[j]) for j in ids]
            rids.sort(reverse=True, key=lambda x: x[1])
            knn_i = [(int(j), dm_i[j]) for (j, _) in rids if j != i]
            ns.append(knn_i)
        t3 = time()
        print('TIME cdist:', round(t2 - t1, 2), 'sorted knns:', round(t3 - t2))
        return ns

    def get_sents(self):
        return from_pickle(self.cache())[0]

    def cluster(self, k=None):
        # Initialize and fit the KMeans model
        sents, embeddings = from_pickle(self.cache())
        if k is None: k = int(len(sents) ** 0.55)
        embeddings = np.array(embeddings)
        sents = np.array(sents)
        kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++', n_init="auto")
        kmeans.fit(embeddings)

        # Get the cluster labels for each data point
        labels = kmeans.labels_

        # Get the coordinates of the cluster centers
        # cluster_centers = kmeans.cluster_centers_

        representative_sentences = []
        for i in range(k):
            cluster_indices = np.where(labels == i)[0]
            # Indices of sentences in the current cluster
            cluster_embeddings = embeddings[cluster_indices]

            # Sentence embeddings for the current cluster
            centroid = cluster_embeddings.mean(axis=0)
            # Calculate the centroid of the cluster
            closest_sentence_index = np.argmin(np.linalg.norm(cluster_embeddings - centroid, axis=1))
            representative_sentence = sents[cluster_indices[closest_sentence_index]]
            cluster_sents = [s for s in sents[cluster_indices] if s != representative_sentence]
            representative_sentences.append((representative_sentence, cluster_sents))

        return representative_sentences

    def __call__(self, quest, top_k):
        return self.query(quest, top_k)

    def dollar_cost(self):
        # if self.LOCAL_LLM: return 0.0
        return self.total_toks * 0.0004 / 1000
