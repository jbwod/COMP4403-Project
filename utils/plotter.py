import matplotlib.pyplot as plt
import networkx as nx

ROLES = {"seeder": "blue", "leecher": "green"}

def draw_graph(graph, total_pieces=None):
    """Draws the given graph using matplotlib."""
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)

    node_colors = []
    for node in graph.nodes():
        role = graph.nodes[node].get("role")
        node_colors.append(ROLES.get(role, 'lightblue'))


    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500)
    plt.title("P2P Network")
    if total_pieces is not None:
        for node in graph.nodes():
            pieces = graph.nodes[node].get("file_pieces", set())
            num_pieces = len(pieces)
            x, y = pos[node]
            plt.text(x, y + 0.15, f"{num_pieces}/{total_pieces}", 
                     ha='center', va='bottom', fontsize=8)
    if any(graph.nodes(data=True)):
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=role) for role, color in ROLES.items()]
        plt.legend(handles=handles, title="Agent Types")
    plt.show()