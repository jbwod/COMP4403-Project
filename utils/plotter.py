import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(graph):
    """Draws the given graph using matplotlib."""
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title("Graph")
    plt.show()