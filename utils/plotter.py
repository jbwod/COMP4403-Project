import matplotlib.pyplot as plt
import networkx as nx

ROLES = {"seeder": "blue", "leecher": "green", "free_rider": "red", "altruist": "orange"}





def draw_graph(graph):
    """Draws the given graph using matplotlib."""
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)

    node_colors = []
    for node in graph.nodes():
        role = graph.nodes[node].get("role")
        node_colors.append(ROLES.get(role, 'lightblue'))


    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500)
    plt.title("Graph")
    if any(graph.nodes(data=True)):
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=role) for role, color in ROLES.items()]
        plt.legend(handles=handles, title="Agent Types")
    plt.show()