import os
import pickle
import json
from config import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

locations=('ROOT','CACHES','DATA','OUT')
def PARAMS():
    """
    config params, easy to propagate as other objects' attributes
    simply by applying it to them
    """

    d = dict(
        TRACE=0,
        ROOT="../STATE/",
        CACHES="caches/",
        OUT='out/',
        DATA='data/',
        model="gpt-3.5-turbo",
        temperature=0.2,
        n=1,
        max_toks=4000
    )
    md=dict((k, d[locations[0]] + v) for (k, v) in d.items() if k in locations[1:])
    return Mdict(**{**d,**md})


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


def jp(x):
    print(json.dumps(x, indent=2))


def xp(xs):
    for x in xs:
        print(x)


if __name__ == "__main__":
    print(PARAMS())
