import math
import random
import networkx as nx

def erdos_renyi_graph(n: int, p: float) -> nx.Graph:
    """ER Graph generator ensuring the graph is connected."""
    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            return G
