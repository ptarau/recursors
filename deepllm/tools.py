# helpers with text cleaning

def spacer(text):
    return ' '.join(text.split())


def de_itemize(x):
    assert isinstance(x, str), x
    x = x
    if x and x[0].isdigit and x[1] == ".":
        r = x[2:]
    elif x and x[0].isdigit and x[1].isdigit and x[2] == ".":
        r = x[3:]
    elif x and x[0] == '-':
        r = x[1:]
    else:
        r = x
    r = r.replace('"', '').replace("'", ' ').strip()
    return r


def clean_up(xs):
    xs = dict((de_itemize(x), True) for x in xs if 8 < len(x) < 2000)
    return list(xs)


def to_list(gs):
    return list(to_stream(gs))


def to_stream(gs):
    while gs:
        g, gs = gs
        yield g


def from_list(xs):
    xs = reversed(xs)
    gs = ()
    for g in xs:
        gs = g, gs
    return gs


def in_stack(newg, gs):
    while gs:
        g, gs = gs
        if g == newg:
            return True
    return False


def to_text(xs):
    return ".\n".join(xs) + "."


def from_text(text):
    lines = text.split("\n")
    res = []
    for line in lines:
        if len(line) < 4: continue
        line = line.strip()
        if line.endswith('?') or line.endswith('.'):
            line = line[0:-1]
        res.append(line)
    return res


def file2string(fname):
    with open(fname, 'r') as f:
        return f.read()
