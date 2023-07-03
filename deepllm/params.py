import os
import pickle
import json
from .configurator import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

IS_LOCAL_LLM = [False]

GPT_PARAMS = dict(
    TRACE=0,
    ROOT="../STATE/",
    CACHES="caches/",
    DATA="data/",
    OUT='out/',
    model="gpt-3.5-turbo",
    emebedding_model="text-embedding-ada-002",
    temperature=0.2,
    n=1,
    max_toks=4000,
    TOP_K=3,
    #LOCAL_LLM=IS_LOCAL_LLM[0]
)

LOCAL_PARAMS = dict(
    TRACE=0,
    ROOT="../STATE_LOCAL/",
    CACHES="caches/",
    OUT='out/',
    DATA='data/',
    model="vicuna-7b-v1.3",
    emebedding_model="vicuna-7b-v1.3",
    temperature=0.2,
    n=1,
    max_toks=2000,
    TOP_K=3,
    API_BASE="http://u.local:8000/v1",  # replace with where the server is
    # API_BASE = "http://localhost:8000/v1" # if on the same machine
    #LOCAL_LLM=IS_LOCAL_LLM[0]
)


def PARAMS():
    """
    config params, easy to propagate as other objects' attributes
    simply by applying it to them
    """

    LOCAL_LLM = IS_LOCAL_LLM[0]

    locations = ['ROOT', 'CACHES', 'DATA', 'OUT']

    if not LOCAL_LLM:
        d = GPT_PARAMS
    else:
        import openai
        openai.api_key = "EMPTY"
        openai.api_base = LOCAL_PARAMS['API_BASE']
        d = LOCAL_PARAMS

    ld = dict((k, d[locations[0]] + v) for (k, v) in d.items() if k in locations[1:])
    # by applying the Mdict of md and d (a callable) to an instance
    # it overrides its attributes with the ones collected from d and ld
    attribute_overrider = Mdict(**{**d, **ld})
    return attribute_overrider


def spacer(text):
    return ' '.join(text.split())


def ensure_path(fname):
    """
    makes sure path to directory and directory exist
    """
    if '/' not in fname: return
    d, _ = os.path.split(fname)
    os.makedirs(d, exist_ok=True)


def exists_file(fname):
    """tests  if it exists as file or dir """
    return os.path.exists(fname)


def remove_file(fname):
    return os.remove(fname)


def to_json(obj, fname, indent=2):
    """
    serializes an object to a json file
    assumes object made of array and dicts
    """
    ensure_path(fname)
    with open(fname, "w") as outf:
        json.dump(obj, outf, indent=indent)


def from_json(fname):
    """
    deserializes an object from a json file
    """
    with open(fname, "rt") as inf:
        obj = json.load(inf)
        return obj


def to_pickle(obj, fname):
    """
    serializes an object to a .pickle file
    """
    ensure_path(fname)
    with open(fname, "wb") as outf:
        pickle.dump(obj, outf)


def from_pickle(fname):
    """
    deserializes an object from a pickle file
    """
    with open(fname, "rb") as inf:
        return pickle.load(inf)


def jp(x):
    print(json.dumps(x, indent=2))


def xp(xs):
    for x in xs:
        print(x)
