import math
import random
import networkx as nx
from typing import Dict

def erdos_renyi_graph(n: int, p: float) -> nx.Graph:
    """ER Graph generator ensuring the graph is connected."""
    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            return G
        


def BA_graph(nodes, edges, seed, weighted, r):
    '''BA graph with weighted and unweighted edges. Random = true means randomly weighted edges'''
    # Rule 1: download unit 1 ud = 1 MB/s, upload unit uu = r*ud for r<1. 
    #         Total weight/bandwith ut = ud + uu = ud(1+r)
    # Rule 2: We use ut for bandwith / weight of edges, and ut will be randomised and assigned directly.

    # Setting bounds
    lower_ut = 5
    uppder_ut = 100

    # Generate le BA graph, unweighted
    graph = nx.barabasi_albert_graph(n=nodes, m=edges, seed=seed)

    # modifying unweighted graph to weighted 
    if weighted == True:
        weights = lambda u, v: random.randint(lower_ut, uppder_ut)
        for u,v in graph.edges():
            graph[u][v]['weight'] = weights(u,v)
        info = Get_edge_info(graph,True,r)
        return graph, info
    else:
        info = Get_edge_info(graph,False,r)
        return graph, info

def Get_edge_info(graph, weighted,r):
    # collect some info from any graph
    edge_info = {
                "edge_ID": [],
                "bandwidth_total": [],
                "bandwidth_download": [],
                "bandwidth_upload": [],
                "node_1": [],
                "node_2": []
                }
    # syntax is, index = index of enumerated edges, 
    #            node_1 = source node, 
    #            node_2 = target node,
    #            data = dictionary of edge attributes
    for index, (node_1, node_2, data) in enumerate(graph.edges(data=True)):
        bandwidth = data.get('weight', None)

        # Compute download/upload only if weighted
        if weighted and bandwidth is not None:
            download = round(bandwidth / (1 + r),1)
            upload = bandwidth - download
        else:
            download = None
            upload = None

        # Append all info
        edge_info["edge_ID"].append(index)
        edge_info["bandwidth_total"].append(bandwidth)
        edge_info["bandwidth_download"].append(download)
        edge_info["bandwidth_upload"].append(upload)
        edge_info["node_1"].append(node_1)
        edge_info["node_2"].append(node_2)
    return edge_info


###################### All code below are for debugging, e.g. plotting graph for checking above code #################

import matplotlib.pyplot as plt

BAA, info = BA_graph(10, 2, 42, False,0.5)
print(info)

def plot_graph_with_bandwidth(graph, edge_info):
    """
    Plots the graph with annotated edge labels showing weight, download, and upload speeds.
    """
    pos = nx.spring_layout(graph, seed=42)

    # Build edge label dictionary
    edge_labels = {}
    for i in range(len(edge_info['edge_ID'])):
        u = edge_info['node_1'][i]
        v = edge_info['node_2'][i]
        weight = edge_info['bandwidth_total'][i]
        download = edge_info['bandwidth_download'][i]
        upload = edge_info['bandwidth_upload'][i]

        label = f"w:{weight}, ↓:{round(download,2)}, ↑:{round(upload,2)}" if weight is not None else "unweighted"
        edge_labels[(u, v)] = label

    # Plot
    plt.figure(figsize=(12, 10))
    nx.draw(graph, pos, node_size=300, with_labels=True, font_size=10, edge_color='gray')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Graph with Bandwidth, Download, and Upload Speeds")
    plt.tight_layout(pad=2.0)
    plt.show()

plot_graph_with_bandwidth(BAA,info)

