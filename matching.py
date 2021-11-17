from collections import defaultdict
from itertools import chain
import numpy as np
import networkx as nx
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
    num_nodes = len(
        set(node for node, _ in p) |
        set(node for node, _ in q)
    )

    row = defaultdict(lambda: len(row))

    X = np.zeros((2, num_nodes))

    for i, nodes in enumerate([p, q]):
        for node_id, weight in nodes:
            X[i, row[node_id]] = weight

    normalize(X, axis=1, norm="l1", copy=False)

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
                u = f"0 {module1}"
                v = f"1 {module2}"
                
                B.add_node(u, bipartite=0, module=module1)
                B.add_node(v, bipartite=1, module=module2)
                B.add_edge(u, v, weight=weight)
    
    return B


def match(P, Q, threshold=None):
    """
    Match modules in partitions using pairwise similarity
    
    Parameters
    ----------
    P, Q
        dict with module ids as keys and list of tuples on the form (id, weight)
       
    Returns
    -------
    M
        nx.Graph
    """
    S, row, col = similarity(P, Q)
    B = create_bipartite_graph(S, row, col)
    
    M = nx.Graph()
    
    for node, data in B.nodes.data(True):
        M.add_node(node, **data)
    
    for source in B.nodes:
        target, data = max(B[source].items(), key=lambda node: node[1]["weight"])
        if threshold is None or data["weight"] > threshold:
            M.add_edge(source, target, weight=data["weight"])
    
    return M


def color_map(M, colors, cmap=None):
    cmap_nodes = color_graph(M, colors, cmap=cmap)
    cmap = defaultdict(dict)
    
    for node, color in cmap_nodes.items():
        partition, module = map(int, node.split())
        cmap[partition][module] = color
    
    return dict(cmap)
    

def color_graph(M, colors, cmap=None):
    cmap_nodes = {}

    if cmap is not None:
        for partition, partition_map in cmap.items():
            for node, color in partition_map.items():
                key = f"{partition} {node}"
                cmap_nodes[key] = color

    added = set(cmap_nodes.keys())
    remaining = set(M.nodes) - added
    i = 0

    for source in chain(added, remaining):
        if source not in cmap_nodes:
            current = colors[i]
            cmap_nodes[source] = current
            i += 1
        else:
            current = cmap_nodes[source]

        for target in M[source]:
            cmap_nodes[target] = current

    nx.set_node_attributes(M, cmap_nodes, "color")

    return cmap_nodes
