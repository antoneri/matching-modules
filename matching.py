from collections import defaultdict
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.preprocessing import normalize


def js_divergence(p: np.array, q: np.array):
    # Jensen-Shannon divergence between p and q
    mix = 0.5 * (p + q)
    jsd = 0.5 * entropy(p, mix, base=2) + 0.5 * entropy(q, mix, base=2)
    
    if jsd < 0 or jsd > 1 + 1e-5:
        raise RuntimeWarning("JSD out of bounds")

    if jsd > 1:
        jsd = 1.0
        
    return jsd


def feature_matrix(p, q):
    num_nodes = len(
        set(node[0] for node in p) |
        set(node[0] for node in q)
    )

    row = defaultdict(lambda: len(row))

    X = np.zeros((2, num_nodes))

    for i, nodes in enumerate([p, q]):
        for node_id, weight in nodes:
            X[i, row[node_id]] = weight

    normalize(X, axis=1, norm="l1", copy=False)

    return X


def js_similarity(p, q):
    """
    Jensen-Shannon similarity between p and q
    
    Parameters
    ----------
    p, q
        iterables of tuples on the form (id, weight)
    
    Returns
    -------
    similarity
        the similarity between p and q
    """
    X = feature_matrix(p, q)
    return 1 - js_divergence(X[0], X[1])


def similarity(P, Q):
    """
    Pairwise similarity between modules in partitions
    
    Parameters
    ----------
    P, Q
        dict with module ids as keys and list of tuples on the form (id, weight)
       
    Returns
    -------
    S
        matrix with the (i,j)th entry being the similarity between module i and j
    row
        mapping from module ids in P to row index
    col
        mapping from module ids in Q to col index
    """
    S = np.zeros((len(P), len(Q)))
    
    row = defaultdict(lambda: len(row))
    col = defaultdict(lambda: len(col))
    
    for module1, p in P.items():
        for module2, q in Q.items():
            i = row[module1]
            j = col[module2]
            S[i, j] = js_similarity(p, q)
    
    return S, dict(row), dict(col)


def create_bipartite_graph(S, row, col):
    row_to_module = {row: module for module, row in row.items()}
    col_to_module = {col: module for module, col in col.items()}
    
    B = nx.Graph()
    
    for i, row in enumerate(S):
        module1 = row_to_module[i]
        
        for j, weight in enumerate(row):
            module2 = col_to_module[j]
            
            if weight > 0.0:
                u = f"P {module1}"
                v = f"Q {module2}"
                
                B.add_node(u, bipartite=0, module=module1)
                B.add_node(v, bipartite=1, module=module2)
                B.add_edge(u, v, weight=weight)
    
    return B


def match(P, Q):
    """
    Match modules in partitions using pairwise similarity
    
    Parameters
    ----------
    P, Q
        dict with module ids as keys and list of tuples on the form (id, weight)
       
    Returns
    -------
    M
        nx.Graph with the 
    """
    S, row, col = similarity(P, Q)
    B = create_bipartite_graph(S, row, col)
    
    M = nx.Graph()
    
    for node, bipartite in B.nodes.data("bipartite"):
        M.add_node(node, bipartite=bipartite)
    
    for source in B.nodes:
        target, data = max(B[source].items(), key=lambda node: node[1]["weight"])
        M.add_edge(source, target, weight=data["weight"])
    
    return M


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


def color_graph(G, colors):
    node_to_color = {}
    current = 0

    for source in G.nodes:
        if source not in node_to_color:
            current_color = colors[current]
            node_to_color[source] = current_color
            current += 1
        else:
            current_color = node_to_color[source]

        for target in G[source]:
            node_to_color[target] = current_color

    nx.set_node_attributes(G, node_to_color, "color")
