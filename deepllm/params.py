import os
import shutil
import pickle
import json
import openai
from deepllm.configurator import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

IS_LOCAL_LLM = [False]

# LOCAL_MODEL="vicuna-7b-v1.5"
LOCAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LOCAL_URL = "http://u.local:8000/v1"  # replace with where the server is

FORCE_TRACE=1
FORCE_SBERT=0

GPT_PARAMS = dict(
    TRACE=FORCE_TRACE,
    TO_SVOS=False,
    ROOT_="./STATE/",
    ROOT="./STATE_SMARTER/",
    CACHES="caches/",
    DATA="data/",
    OUT='out/',
    # model="gpt-3.5-turbo",
    # model="gpt-4",
    model='gpt-4-turbo-preview',
    emebedding_model="text-embedding-3-large",
    temperature=0.2,
    n=1,
    max_toks=12000,
    TOP_K=3,
    API_BASE="https://api.openai.com/v1",
    LOCAL_LLM=IS_LOCAL_LLM[0]

)

LOCAL_PARAMS = dict(
    TRACE=FORCE_TRACE,
    TO_SVOS=False,
    ROOT="./STATE_LOCAL/",
    CACHES="caches/",
    OUT='out/',
    DATA='data/',

    model=LOCAL_MODEL,
    API_BASE=LOCAL_URL,

    # emebedding_model="vicuna-7b-v1.5",
    emebedding_model="text-embedding-3-large",

    temperature=0.2,
    n=1,
    max_toks=12000,
    TOP_K=3,

    LOCAL_LLM=IS_LOCAL_LLM[0]
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
        d = LOCAL_PARAMS

    ld = dict((k, d[locations[0]] + v) for (k, v) in d.items() if k in locations[1:])
    # by applying the Mdict of md and prompters (a callable) to an instance
    # it overrides its attributes with the ones collected from prompters and ld
    attribute_overrider = Mdict(**{**d, **ld})
    return attribute_overrider


API_KEY = [os.getenv("OPENAI_API_KEY")]


def set_openai_api_key(key):
    assert key
    assert len(key) > 40
    API_KEY[0] = key
    return key


def ensure_openai_api_key():
    if IS_LOCAL_LLM[0]:
        return "EMPTY"
    else:
        return API_KEY[0]


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


def remove_dir(dname):
    if exists_file(dname):
        shutil.rmtree(dname)


def copy_file(src, dst):
    return shutil.copyfile(src, dst)


def clear_caches():
    dirs=[
        GPT_PARAMS['ROOT'],
        GPT_PARAMS['ROOT_'],
        LOCAL_PARAMS['ROOT']
    ]
    for d in dirs:
        remove_dir(d)
    return dirs


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


def jpp(x):
    print(json.dumps(x, indent=2))


def xp(xs):
    for x in xs:
        print(x)


def tprint(*args, **kwargs):
    if PARAMS().TRACE:
        print(*args, **kwargs)
