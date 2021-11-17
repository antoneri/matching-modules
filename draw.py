import math
import networkx as nx
import matplotlib.pyplot as plt


def draw_match(G, ax=None, pos=None):
    left = [node for node, bipartite in G.nodes.data("bipartite") if bipartite == 0]

    if pos is None:
        pos = nx.bipartite_layout(G, left)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    edgelist = nx.get_edge_attributes(G, "weight")

    colors = list(nx.get_node_attributes(G, "color").values())

    if len(colors) == 0:
        colors = "#eee"

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000, node_color=colors)

    nx.draw_networkx_labels(G, pos, ax=ax, font_color="black", font_weight="bold")

    nx.draw_networkx_edges(G,
                           pos,
                           ax=ax,
                           edgelist=edgelist.keys(),
                           alpha=0.5,
                           width=[2 * math.exp(w) for w in edgelist.values()])

    return pos

