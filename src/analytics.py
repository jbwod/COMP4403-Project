import networkx as nx

def clustering_co(Graph):
    cluster = nx.average_clustering(Graph)
    return cluster

def path_length(Graph):
    path_length = nx.average_shortest_path_length(Graph)
    return path_length

    
    