from pyvis.network import Network

import webbrowser
import os

def visualize_rels(svos, fname='rel_graph', show=True):

    net = Network(directed=True, height="1200px", )
    shape = 'box'

    es=set()

    ns = set(x for (s,_,o) in svos for x in (s,o))

    for n in ns:
        net.add_node(n, shape=shape,mass=4,color='green')

    def add(x, v, y):
        if (x, v, y) not in es:
            es.add((x,v,y))
            net.add_edge(x, y, label=v, color='blue')

    for s, v, o in svos:
        add(s, v, o)

    net.toggle_physics(True)
    hfile = fname + '.html'
    net.show(hfile, notebook=False)
    url = "file://" + os.path.abspath(hfile)
    if show: browse(url)
    return url,hfile


def browse(url):
    print('BROWSING:', url)
    return webbrowser.open(url)


if __name__ == "__main__":
    visualize_rels([('a', 'v', 'b'), ('b', 'u', 'c'), ('c', 'w', 'a')])
