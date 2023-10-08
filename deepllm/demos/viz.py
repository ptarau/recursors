import networkx as nx
import matplotlib.pyplot as plt
from deepllm.api import *


def vizrun(lim=1):
    for result in run_rater(initiator='Artificial General Intelligence', prompter=sci_prompter, lim=lim,
                            threshold=0.30):
        kind, vals = result
        if kind == 'CLAUSES':
            clauses = vals
        elif kind == 'MODEL':
            model = vals
        else:
            jpp(result)

    css = [(h, xs) for (h, xss) in clauses.items() for xs in xss]

    if lim > 1:
        model = dict(zip(model, range(len(model))))
        nss = [(model[h], [model[b] for b in bs]) for (h, bs) in css]
        g = to_horn_graph(nss)
    else:
        g = to_horn_graph(css)
    print(g)
    draw(g)


def to_horn_graph(css, ics=None):
    g = nx.DiGraph()
    for i, c in enumerate(css):
        if isinstance(c, tuple):
            h, bs = c
            if bs == []:
                g.add_edge(True, h, clause=i)

            else:
                for b in bs:
                    g.add_edge(b, h, clause=i)
            if ics is not None:
                for ic in ics:
                    g.add_edge(ic, 'false', clause=i)
        else:
            g.add_edge(True, c, clause=i)

    return g


def draw(G, edge_label='clause'):
    """
    draws (directed) graph using node names as node labels
    and give edge_label for labeling edges
    """
    # pos = nx.spring_layout(G)
    pos = nx.nx_agraph.graphviz_layout(G)

    plt.figure()
    nx.draw(
        G, pos, edge_color='black', width=1, linewidths=2,
        node_size=500, node_color='grey', alpha=0.9,
        labels={node: node for node in G.nodes()},
        arrows=True,
    )
    edge_labels = [((x, y), G[x][y][edge_label]) for (x, y) in G.edges()]
    nx.draw_networkx_edge_labels(
        G, pos,
        font_color='black',
        edge_labels=dict(edge_labels)
    )
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    vizrun()
