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
class TransferConfig:
    """Configuration for transfers."""
    arrow_color: str = 'red'
    arrow_width: int = 2
    arrow_alpha: float = 0.7
    label_bg_color: str = 'yellow'
    label_alpha: float = 0.7

class GraphPlotter:    
    def __init__(self, plot_config: PlotConfig = None, transfer_config: TransferConfig = None, 
                 save_images: bool = False, output_dir: str = None):
        self.plot_config = plot_config or PlotConfig()
        self.transfer_config = transfer_config or TransferConfig()
        self.save_images = save_images
        self.output_dir = output_dir or self.create_output_directory()
        self.image_counter = 0
    
    def create_output_directory(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        return filepath
    
    def get_node_colors(self, graph: nx.Graph, transfers: List[Dict] = None) -> List[str]:
        """Get colors for nodes based on their roles."""
        colors = []
        for node in graph.nodes():
            role = graph.nodes[node].get("role")
            
            # both sending and recieving?
            is_hybrid = False
            if transfers:
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
            plt.text(x, y + 0.15, f"{num_pieces}/{total_pieces}", 
                     ha='center', va='bottom', fontsize=8)
    
    def draw_transfer_arrows(self, transfers: List[Dict], pos: Dict):
        """Draw arrows showing transfers."""
        for transfer in transfers:
            from_node = transfer["from"]
            to_node = transfer["to"]
            piece = transfer["piece"]
            
            if from_node in pos and to_node in pos:
                x1, y1 = pos[from_node]
                x2, y2 = pos[to_node]
                
                plt.arrow(x1, y1, x2 - x1, y2 - y1, 
                          color=self.transfer_config.arrow_color, 
                          width=0.01, alpha=self.transfer_config.arrow_alpha, 
                          length_includes_head=True, head_width=0.05)
                plt.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, f'P{piece}', 
                         ha='center', va='bottom', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    def create_legend(self, graph: nx.Graph, show_transfers: bool = False):
        """Create legend for the plot."""
        if not self.plot_config.show_legend or not any(graph.nodes(data=True)):
            return
            
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=10, label=role.title()) 
            for role, color in ROLES.items()
        ]
        
        if show_transfers:
            handles.append(plt.Line2D([0], [0], color=self.transfer_config.arrow_color, 
                                    lw=self.transfer_config.arrow_width, label='Transfers'))
        
        plt.legend(handles=handles, title="Legend")
    
    def create_title(self, round_num: Optional[int] = None, transfers: Optional[List] = None) -> str:
        """Create title for the plot."""
        title = "P2P Network"
        if round_num is not None:
            title += f" - Round {round_num}"
        if transfers:
            title += f" ({len(transfers)} transfers)"
        return title
    
    def draw_base_graph(self, graph: nx.Graph, total_pieces: Optional[int] = None, 
                       title: Optional[str] = None) -> None:
        """Draw the base graph without transfers."""
        plt.figure(figsize=self.plot_config.figure_size)
        pos = self.get_node_positions(graph)
        node_colors = self.get_node_colors(graph)
        
        nx.draw(graph, pos, with_labels=self.plot_config.show_labels, 
                node_color=node_colors, edge_color=self.plot_config.edge_color, 
                node_size=self.plot_config.node_size)
        
        if total_pieces is not None:
            self.draw_piece_counters(graph, pos, total_pieces)
        
        plt.title(title or self.create_title())
        self.create_legend(graph)
        
        # Save if enabled
        if self.save_images:
            saved_path = self.save_image()
            print(f"graph image: {saved_path}")
        
        plt.show()
    
    def draw_with_transfers(self, graph: nx.Graph, total_pieces: Optional[int] = None, 
                          transfers: Optional[List[Dict]] = None, 
                          round_num: Optional[int] = None) -> None:
        """Draw graph with transfer visual."""
        plt.figure(figsize=self.plot_config.figure_size)
        pos = self.get_node_positions(graph)
        node_colors = self.get_node_colors(graph, transfers)
        
        nx.draw(graph, pos, with_labels=self.plot_config.show_labels, 
                node_color=node_colors, edge_color=self.plot_config.edge_color, 
                node_size=self.plot_config.node_size)
        
        if total_pieces is not None:
            self.draw_piece_counters(graph, pos, total_pieces)
        
        if transfers:
            self.draw_transfer_arrows(transfers, pos)
        
        plt.title(self.create_title(round_num, transfers))
        self.create_legend(graph, show_transfers=bool(transfers))
        
        # Save if enabled
        if self.save_images:
            filename = f"round_{round_num:03d}.png" if round_num is not None else None
            saved_path = self.save_image(filename)
            print(f"graph image: {saved_path}")
        
        plt.show()

def draw_graph(graph: nx.Graph, total_pieces: Optional[int] = None, 
               save_images: bool = False) -> None:
    """Draws the given graph using matplotlib."""
    plotter = GraphPlotter(save_images=save_images)
    plotter.draw_base_graph(graph, total_pieces)

def draw_graph_with_transfers(graph: nx.Graph, total_pieces: Optional[int] = None, 
                            transfers: Optional[List[Dict]] = None, 
                            round_num: Optional[int] = None, 
                            save_images: bool = False) -> None:
    plotter = GraphPlotter(save_images=save_images)
    plotter.draw_with_transfers(graph, total_pieces, transfers, round_num)