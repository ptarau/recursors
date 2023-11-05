from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
import openai
from deepllm.params import to_pickle, from_pickle, PARAMS


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
        # print('!!!', rids)
        answers = [(sents[i], dm[i]) for (i, _) in rids]
        return answers

    def cluster(self, k=None):
        # Initialize and fit the KMeans model
        sents, embeddings = from_pickle(self.cache())
        if k is None: k=int(len(sents)**0.55)
        embeddings = np.array(embeddings)
        sents=np.array(sents)
        kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++', n_init="auto")
        kmeans.fit(embeddings)

        # Get the cluster labels for each data point
        labels = kmeans.labels_

        # Get the coordinates of the cluster centers
        cluster_centers = kmeans.cluster_centers_

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
            representative_sentences.append((representative_sentence,cluster_sents))

        return representative_sentences

    def __call__(self, quest, top_k):
        return self.query(quest, top_k)

    def dollar_cost(self):
        if self.LOCAL_LLM: return 0.0
        return self.total_toks * 0.0004 / 1000
