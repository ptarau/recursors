from deepllm.embedders import Embedder


def test_embedders():
    e = Embedder(cache_name='embedder_test')
    sents = [
        "The dog barks to the moon",
        "The cat sits on the mat",
        "The phone rings",
        "The rocket explodes",
        "The cat and the dog sleep"
    ]
    e.store(sents)
    q = 'Who sleeps on the mat?'
    rs = e(q, 2)
    for r in rs: print(r)
    print('COST:', e.dollar_cost())
    print('TIMES:',e.times)
    return True


if __name__ == "__main__":
    assert test_embedders()
