import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as patches

ROLES = {"seeder": "blue", "leecher": "green"}

def draw_graph(graph, total_pieces=None):
    """Draws the given graph using matplotlib."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)

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


def draw_graph_with_transfers(graph, total_pieces=None, transfers=None, round_num=None):

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)

    node_colors = []
    for node in graph.nodes():
        role = graph.nodes[node].get("role")
        node_colors.append(ROLES.get(role, 'lightblue'))

    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500)

    # Add piece counters
    if total_pieces is not None:
        for node in graph.nodes():
            pieces = graph.nodes[node].get("file_pieces", set())
            num_pieces = len(pieces)
            x, y = pos[node]
            plt.text(x, y + 0.15, f"{num_pieces}/{total_pieces}", 
                     ha='center', va='bottom', fontsize=8)

    # Draw transfer arrows
    if transfers:
        for i, transfer in enumerate(transfers):
            from_node = transfer["from"]
            to_node = transfer["to"]
            piece = transfer["piece"]
            
            if from_node in pos and to_node in pos:
                x1, y1 = pos[from_node]
                x2, y2 = pos[to_node]
                

                plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
                

                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                plt.text(mid_x, mid_y + 0.05, f'P{piece}', 
                        ha='center', va='bottom', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))


    title = "P2P Network"
    if round_num is not None:
        title += f" - Round {round_num}"
    if transfers:
        title += f" ({len(transfers)} transfers)"
    plt.title(title)

    if any(graph.nodes(data=True)):
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=role) for role, color in ROLES.items()]
        handles.append(plt.Line2D([0], [0], color='red', lw=2, label='Transfers'))
        plt.legend(handles=handles, title="Legend")
    
    plt.show()