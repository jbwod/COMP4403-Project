import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from datetime import datetime


ROLES = {"seeder": "blue", "leecher": "green", "hybrid": "purple"}
DEFAULT_FIGURE_SIZE = (10, 8)
DEFAULT_LAYOUT_SEED = 42

@dataclass
class PlotConfig:
    figure_size: Tuple[int, int] = DEFAULT_FIGURE_SIZE
    layout_seed: int = DEFAULT_LAYOUT_SEED
    node_size: int = 500
    edge_color: str = 'gray'
    show_labels: bool = True
    show_legend: bool = True


@dataclass
class GossipConfig:
    """Configuration for gossip messages."""
    query_color: str = 'purple'
    hit_color: str = 'green'
    arrow_width: int = 1
    arrow_alpha: float = 0.6
    query_style: str = 'dashed'
    hit_style: str = 'solid'
    label_bg_color: str = 'lightblue'
    label_alpha: float = 0.8

class GraphPlotter:    
    def __init__(self, plot_config: PlotConfig = None, gossip_config: GossipConfig = None,
                 save_images: bool = False, output_dir: str = None):
        self.plot_config = plot_config or PlotConfig()
        self.gossip_config = gossip_config or GossipConfig()
        self.save_images = save_images
        self.output_dir = output_dir or self.create_output_directory()
        self.image_counter = 0
    
    def create_output_directory(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f"data/simulation_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def save_image(self, filename: str = None) -> str:
        if not self.save_images:
            return None
        
        if filename is None:
            self.image_counter += 1
            filename = f"graph_{self.image_counter:03d}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=500, bbox_inches='tight')
        return filepath
    
    def get_node_colors(self, graph: nx.Graph, transfers: List[Dict] = None) -> List[str]:
        """Get colors for nodes based on their roles."""
        colors = []
        for node in graph.nodes():
            role = graph.nodes[node].get("role")
            
            # Check if node is hybrid role or both sending and receiving in transfers
            is_hybrid = (role == "hybrid")
            if transfers and not is_hybrid:
                sending = any(t["from"] == node for t in transfers)
                receiving = any(t["to"] == node for t in transfers)
                is_hybrid = sending and receiving
            
            if is_hybrid:
                colors.append(ROLES["hybrid"])
            else:
                colors.append(ROLES.get(role, 'lightblue'))
        
        return colors
    
    def get_node_positions(self, graph: nx.Graph) -> Dict:
        """Get node positions using spring."""
        return nx.spring_layout(graph, seed=self.plot_config.layout_seed)
    
    def draw_piece_counters(self, graph: nx.Graph, pos: Dict, total_pieces: int):
        """Draw piece counters above nodes."""
        for node in graph.nodes():
            pieces = graph.nodes[node].get("file_pieces", set())
            num_pieces = len(pieces)
            x, y = pos[node]
            plt.text(x, y + 0.1, f"{num_pieces}/{total_pieces}", 
                     ha='center', va='bottom', fontsize=8)
    
    def draw_gossip_transfer_line(self, from_node: int, to_node: int, piece: int, pos: Dict, query_uuid: str = None) -> None:
        """Draw a transfer line between nodes when a query successfully identifies a source."""
        if from_node not in pos or to_node not in pos:
            return
        
        x1, y1 = pos[from_node]
        x2, y2 = pos[to_node]
        
        # Line instead of arrow
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=4, alpha=0.8, zorder=1)
        
        plt.text((x1 + x2) / 2, (y1 + y2) / 2, f'P{piece}', 
                         ha='center', va='bottom', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        
        # Add transfer direction indicator
        plt.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, f"Transfer", 
                ha='center', va='bottom', fontsize=8, alpha=0.8, zorder=10)
    
    
    def create_legend(self, graph: nx.Graph, show_gossip: bool = False):
        """Create legend for the plot."""
        if not self.plot_config.show_legend or not any(graph.nodes(data=True)):
            return
            
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=10, label=role.title()) 
            for role, color in ROLES.items()
        ]
        
        if show_gossip:
            handles.extend([
                plt.Line2D([0], [0], color=self.gossip_config.query_color, 
                          linewidth=2, linestyle=self.gossip_config.query_style, 
                          label='Query Messages (Purple Dashed)'),
                plt.Line2D([0], [0], color=self.gossip_config.hit_color, 
                          linewidth=2, linestyle=self.gossip_config.hit_style, 
                          label='Hit Messages (Green Solid)'),
                plt.Line2D([0], [0], color='red', linewidth=4, 
                          label='Transfer Lines (Red Thick)')
            ])
        
        plt.legend(handles=handles, title="Legend")
    
    def create_title(self, round_num: Optional[int] = None, transfers: Optional[List] = None) -> str:
        """Create title for the plot."""
        title = "P2P Network"
        if round_num is not None:
            title += f" - Round {round_num}"
        if transfers:
            title += f" ({len(transfers)} transfers)"
        return title
    
    def draw_base_graph(self, graph: nx.Graph, edge_labels: Optional[str] = None, total_pieces: Optional[int] = None, 
                       title: Optional[str] = None) -> None:
        """Draw the base graph without transfers."""
        plt.figure(figsize=self.plot_config.figure_size)
        pos = self.get_node_positions(graph)
        node_colors = self.get_node_colors(graph)
        
        if edge_labels:
            for (u, v), weight in edge_labels.items():
                if graph.has_edge(u, v):
                    graph[u][v]['weight'] = weight
        
        auto_edge_labels = None
        if edge_labels is None:
            labels = {}
            has_any_weight = False
            for u, v, data in graph.edges(data=True):
                w = data.get('weight')
                if w is not None:
                    has_any_weight = True
                    labels[(u, v)] = f"B:{w} MB/s"
            if has_any_weight:
                auto_edge_labels = labels

        nx.draw(graph, pos, with_labels=self.plot_config.show_labels, 
                node_color=node_colors, edge_color=self.plot_config.edge_color, 
                node_size=self.plot_config.node_size)

        nx.draw_networkx_edge_labels(graph, pos, edge_labels=(edge_labels or auto_edge_labels), font_size=8)
        
        if total_pieces is not None:
            self.draw_piece_counters(graph, pos, total_pieces)
        
        plt.title(title or self.create_title())
        self.create_legend(graph)
        
        # Save if enabled
        if self.save_images:
            saved_path = self.save_image()
            print(f"graph image: {saved_path}")
        
        plt.show()
    
    
    
    def draw_gossip_messages(self, graph: nx.Graph, messages: List[Dict], pos: Dict, total_pieces: Optional[int] = None, round_num: Optional[int] = None) -> None:
        """Draw gossip messages."""
        for message in messages:
            if message["type"] == "query":
                self.draw_query_arrow(message, pos)
            elif message["type"] == "hit":
                self.draw_hit_arrow(message, pos)
    
    def draw_query_arrow(self, query: Dict, pos: Dict) -> None:
        """Draw a query message arrow."""
        from_node = query["from_node"]
        to_node = query["to_node"]
        piece = query["piece"]
        ttl = query["ttl"]
        query_uuid = query.get("query_uuid", "")
        origin = query.get("origin", from_node)
        if from_node not in pos or to_node not in pos:
            return
        
        x1, y1 = pos[from_node]
        x2, y2 = pos[to_node]
        
        # Draw dashed purple arrow for queries
        plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', 
                                  color=self.gossip_config.query_color,
                                  linestyle=self.gossip_config.query_style,
                                  linewidth=self.gossip_config.arrow_width,
                                  alpha=self.gossip_config.arrow_alpha))
        # Format: Q: [First 3 letters of uid] PX O:[Origin] TTL:
        uuid_short = query_uuid[:3] if query_uuid else "???"
        label_text = f"Q: {uuid_short} P{piece} O:{origin} TTL:{ttl}"
        plt.text((x1 + x2) / 2, (y1 + y2) / 2, label_text,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.gossip_config.label_bg_color,
                         alpha=self.gossip_config.label_alpha, edgecolor=self.gossip_config.query_color),
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    def draw_hit_arrow(self, hit: Dict, pos: Dict) -> None:
        """Draw a QueryHit message arrow."""
        from_node = hit["from_node"]
        to_node = hit["to_node"]
        piece = hit["piece"]
        query_uuid = hit.get("query_uuid", "")
        hit_node = hit.get("hit_node", from_node)
        origin = hit.get("origin", to_node)
        
        if from_node not in pos or to_node not in pos:
            return
        
        x1, y1 = pos[from_node]
        x2, y2 = pos[to_node]
        
        # Draw solid green arrow
        plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', 
                                  color=self.gossip_config.hit_color,
                                  linestyle=self.gossip_config.hit_style,
                                  linewidth=self.gossip_config.arrow_width,
                                  alpha=self.gossip_config.arrow_alpha))
        
        # Format: H: [First 3 letters of uid] PX O:[Origin] H:[HitNode]
        label_text = f"H: {query_uuid[:3] if query_uuid else '???'} P{piece} O:{origin} H:{hit_node}"
        plt.text((x1 + x2) / 2, ((y1 + y2) / 2 + 0.05), label_text,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.gossip_config.label_bg_color,
                         alpha=self.gossip_config.label_alpha, edgecolor=self.gossip_config.hit_color),
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    def draw_gossip_round(self, graph: nx.Graph, messages: List[Dict], transfers: List[Dict],
                         total_pieces: Optional[int] = None, round_num: Optional[int] = None) -> None:
        """Draw a complete gossip round with messages and transfers."""
        plt.figure(figsize=self.plot_config.figure_size)
        pos = self.get_node_positions(graph)
        node_colors = self.get_node_colors(graph, transfers)
        
        # Draw base graph
        nx.draw(graph, pos, with_labels=self.plot_config.show_labels, 
                node_color=node_colors, edge_color=self.plot_config.edge_color, 
                node_size=self.plot_config.node_size)
        
        if total_pieces is not None:
            self.draw_piece_counters(graph, pos, total_pieces)
        
        # Draw gossip messages
        if messages:
            self.draw_gossip_messages(graph, messages, pos)
        
        # Draw transfer lines for successful queries
        if transfers:
            for transfer in transfers:
                self.draw_gossip_transfer_line(
                    transfer["from"], 
                    transfer["to"], 
                    transfer["piece"], 
                    pos,
                    transfer.get("query_uuid")
                )
        
        title = f"Round {round_num} | Gossip" if round_num else "Gossip"
        if messages:
            query_count = len([m for m in messages if m["type"] == "query"])
            hit_count = len([m for m in messages if m["type"] == "hit"])
            title += f" ({query_count} queries, {hit_count} hits)"
        if transfers:
            title += f" ({len(transfers)} transfers)"
        
        plt.title(title)
        self.create_legend(graph, show_gossip=True)
        
        if self.save_images:
            filename = f"round_{round_num:03d}_gossip.png" if round_num is not None else None
            saved_path = self.save_image(filename)
            print(f"Gossip round image: {saved_path}")
        
        plt.show()
    
  
    
    def draw_gossip_step_by_step(self, graph: nx.Graph, message_rounds: List[List[Dict]], 
                                transfers: List[Dict], total_pieces: Optional[int] = None, 
                                round_num: Optional[int] = None) -> None:
        """
        Draw step-by-step visualization: queries (step 1), hits (step 2), transfers (step 3).
        In step 3, draw the graph without edges and then overlay transfer lines.
        """
        if len(message_rounds) < 3:
            # Pad with empty rounds if needed
            while len(message_rounds) < 3:
                message_rounds.append([])
        
        # Create subplots for each step
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        pos = self.get_node_positions(graph)
        
        # 1. Queries
        ax1 = axes[0]
        plt.sca(ax1)
        
        # Draw base graph
        node_colors = self.get_node_colors(graph)
        nx.draw(graph, pos, with_labels=self.plot_config.show_labels, 
                node_color=node_colors, edge_color=self.plot_config.edge_color, 
                node_size=self.plot_config.node_size)
        
        if total_pieces is not None:
            self.draw_piece_counters(graph, pos, total_pieces)
        
        # Draw queries
        queries = message_rounds[0] if message_rounds[0] else []
        if queries:
            self.draw_gossip_messages(graph, queries, pos)
        
        query_count = len([m for m in queries if m["type"] == "query"])
        ax1.set_title(f"1. Queries ({query_count} queries)")

        # 2. Hits
        ax2 = axes[1]
        plt.sca(ax2)
        
        # Draw base graph
        nx.draw(graph, pos, with_labels=self.plot_config.show_labels, 
                node_color=node_colors, edge_color=self.plot_config.edge_color, 
                node_size=self.plot_config.node_size)
        
        if total_pieces is not None:
            self.draw_piece_counters(graph, pos, total_pieces)
        
        # Draw hits
        hits = message_rounds[1] if message_rounds[1] else []
        if hits:
            self.draw_gossip_messages(graph, hits, pos)
        
        hit_count = len([m for m in hits if m["type"] == "hit"])
        ax2.set_title(f"Step 2: Hits ({hit_count} hits)")
        
        # 3. Transfers
        ax3 = axes[2]
        plt.sca(ax3)
        
        # Draw graph WITHOUT edges, only nodes (since transfers are direct)
        nx.draw(graph, pos, with_labels=self.plot_config.show_labels, 
                node_color=node_colors, edge_color='none', 
                node_size=self.plot_config.node_size)
        
        if total_pieces is not None:
            self.draw_piece_counters(graph, pos, total_pieces)
        
        # Draw transfer lines
        if transfers:
            for transfer in transfers:
                self.draw_gossip_transfer_line(
                    transfer["from"], 
                    transfer["to"], 
                    transfer["piece"], 
                    pos,
                    transfer.get("query_uuid")
                )
        
        ax3.set_title(f"Step 3: Transfers ({len(transfers)} transfers)")
        
        # Overall title
        title = f"Gossip Steps - Round {round_num}" if round_num else "Gossip Steps"
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if self.save_images:
            filename = f"round_{round_num:03d}_gossip_steps.png" if round_num is not None else "gossip_steps.png"
            saved_path = self.save_image(filename)
            print(f"Gossip step-by-step image: {saved_path}")
        
        plt.show()

def draw_graph(graph: nx.Graph, edge_labels: Optional[Dict[Tuple[int, int], float]] = None, total_pieces: Optional[int] = None, 
               save_images: bool = False) -> None:
    """Draws the given graph using matplotlib."""
    plotter = GraphPlotter(save_images=save_images)
    plotter.draw_base_graph(graph, edge_labels=edge_labels, total_pieces=total_pieces)


def draw_gossip_round(graph: nx.Graph, messages: List[Dict], transfers: List[Dict],
                     total_pieces: Optional[int] = None, round_num: Optional[int] = None, 
                     save_images: bool = False) -> None:
    """Draw a complete gossip round with messages and transfers."""
    plotter = GraphPlotter(save_images=save_images)
    plotter.draw_gossip_round(graph, messages, transfers, total_pieces, round_num)


def draw_gossip_flow(graph: nx.Graph, search_result: Dict, total_pieces: Optional[int] = None, 
                    save_images: bool = False) -> None:
    """Draw the flow of a single gossip search showing the path taken and nodes visited."""
    plotter = GraphPlotter(save_images=save_images)
    plotter.draw_gossip_flow(graph, search_result, total_pieces)

def draw_gossip_step_by_step(graph: nx.Graph, message_rounds: List[List[Dict]], 
                           transfers: List[Dict], total_pieces: Optional[int] = None, 
                           round_num: Optional[int] = None, save_images: bool = False) -> None:
    """Draw step-by-step visualization: queries (step 1), hits (step 2), transfers (step 3)."""
    plotter = GraphPlotter(save_images=save_images)
    plotter.draw_gossip_step_by_step(graph, message_rounds, transfers, total_pieces, round_num)