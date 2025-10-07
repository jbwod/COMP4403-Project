import math
import random
import networkx as nx
from typing import Dict

class nxgraph:
    def __init__(self):
        self.graph = None
        self.edge_info = {
            "edge_ID": [],
            "bandwidth_total": [],
            "bandwidth_download": [],
            "bandwidth_upload": [],
            "node_1": [],
            "node_2": []
        }

    def assign_bandwidth_weights(self, graph: nx.Graph, lower_ut: int = 5, upper_ut: int = 100) -> None:
        """Assign random bandwidth weights to all edges in the graph."""
        for u, v in graph.edges():
            graph[u][v]['weight'] = random.randint(lower_ut, upper_ut)

    def ER_graph(self, nodes: int, prob: float, weighted: bool = True, lower_ut: int = 5, upper_ut: int = 100) -> nx.Graph:
        """ER Graph generator ensuring the graph is connected, with optional weighted edges."""
        print('Generating ER graph')
        while True:
            G = nx.erdos_renyi_graph(nodes, prob)
            if nx.is_connected(G):
                if weighted:
                    self.assign_bandwidth_weights(G, lower_ut, upper_ut)
                self.graph = G
                return G

    def BA_graph(self, nodes: int, edges: int, seed: int, weighted: bool, lower_ut: int = 5, upper_ut: int = 100) -> nx.Graph:
        """BA graph with optional weighted edges."""
        print('Generating BA Graph')
        # Rule 1: download unit 1 ud = 1 MB/s, upload unit uu = r*ud for r<1. 
        #         Total weight/bandwith ut = ud + uu = ud(1+r)
        # Rule 2: We use ut for bandwith / weight of edges, and ut will be randomised and assigned directly.
        lower_ut = 5
        upper_ut = 100

        G = nx.barabasi_albert_graph(n=nodes, m=edges, seed=seed)

        if weighted:
            self.assign_bandwidth_weights(G, lower_ut, upper_ut)

        self.graph = G
        return G

    def assign_weights(self, G: nx.Graph = None, lower_ut: int = 5, upper_ut: int = 100) -> nx.Graph:
        """Assign random bandwidth weights to all edges"""
        target = G or self.graph
        self.assign_bandwidth_weights(target, lower_ut, upper_ut)
        if G is None:
            self.graph = target
        return target

    def get_info(self, weighted: bool, r: float) -> Dict:
        """Extract edge info from the graph."""
        if self.graph is None:
            raise ValueError("Graph has not been initialized.")
        # Reset edge
        self.edge_info = {
            "edge_ID": [],
            "bandwidth_total": [],
            "bandwidth_download": [],
            "bandwidth_upload": [],
            "node_1": [],
            "node_2": []
        }

        for index, (node_1, node_2, data) in enumerate(self.graph.edges(data=True)):
            bandwidth = data.get('weight', None)

            if weighted and bandwidth is not None:
                download = round(bandwidth / (1 + r), 1)
                upload = round(bandwidth - download,1)
            else:
                download = None
                upload = None

            self.edge_info["edge_ID"].append(index)
            self.edge_info["bandwidth_total"].append(bandwidth)
            self.edge_info["bandwidth_download"].append(download)
            self.edge_info["bandwidth_upload"].append(upload)
            self.edge_info["node_1"].append(node_1)
            self.edge_info["node_2"].append(node_2)

        return self.edge_info

# ###################### All code below are for debugging, e.g. plotting graph for checking above code #################

# import matplotlib.pyplot as plt
# BAA = nxgraph()
# BAA.BA_graph(10, 2, 42, True)
# #BAA.ER_graph(20,0.1)
# info = BAA.get_info(True,0.5)
# print(info)

# def plot_graph_with_bandwidth(graph, edge_info):
#     """
#     Plots the graph with annotated edge labels showing weight, download, and upload speeds.
#     """
#     pos = nx.spring_layout(graph, seed=42)

#     # Build edge label dictionary
#     edge_labels = {}
#     for i in range(len(edge_info['edge_ID'])):
#         u = edge_info['node_1'][i]
#         v = edge_info['node_2'][i]
#         weight = edge_info['bandwidth_total'][i]
#         download = edge_info['bandwidth_download'][i]
#         upload = edge_info['bandwidth_upload'][i]

#         label = f"w:{weight}, ↓:{round(download,2)}, ↑:{round(upload,2)}" if weight is not None else "unweighted"
#         edge_labels[(u, v)] = label

#     # Plot
#     plt.figure(figsize=(12, 10))
#     nx.draw(graph, pos, node_size=300, with_labels=True, font_size=10, edge_color='gray')
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
#     plt.title("Graph with Bandwidth, Download, and Upload Speeds")
#     plt.tight_layout(pad=2.0)
#     plt.show()

# plot_graph_with_bandwidth(BAA.graph,info)

