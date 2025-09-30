import random
from typing import Dict, Optional

from src.agent import update_agent_completion

import networkx as nx


def simulate_round(G: nx.Graph, total_pieces: int, seed: Optional[int] = None) -> Dict:
    """
    Simo sending pieces to leechers
    """
    rng = random.Random(seed)
    transfers = []
    new_completions = []
    
    #(agents who need pieces)
    leechers = [node for node in G.nodes() 
                if G.nodes[node].get("role") == "leecher" and not G.nodes[node].get("is_complete", False)]
    
    # Collect all planned transfers first (without executing them)
    planned_transfers = []
    
    for leecher in leechers:
        leecher_pieces = G.nodes[leecher].get("file_pieces", set())
        download_capacity = G.nodes[leecher].get("download_capacity", 1)
        
        # Find those who can share
        neighbors = list(G.neighbors(leecher))
        potential_sources = []
        
        for neighbor in neighbors:
            neighbor_pieces = G.nodes[neighbor].get("file_pieces", set())
            # this is the pieces the neighbor has that the leecher doesn't
            available_pieces = neighbor_pieces - leecher_pieces
            if available_pieces:
                potential_sources.append((neighbor, available_pieces))
        
        # Randomly select pieces to download (up to capacity) of the leecher
        pieces_to_download = []
        for source, available_pieces in potential_sources:
            if len(pieces_to_download) >= download_capacity:
                break
            # Randomly select pieces from this source (neighbor)
            source_pieces = list(available_pieces)
            rng.shuffle(source_pieces)
            for piece in source_pieces:
                if len(pieces_to_download) >= download_capacity:
                    break
                pieces_to_download.append((source, piece))
        
        # Plan transfers (don't execute yet)
        for source, piece in pieces_to_download:
            planned_transfers.append({
                "from": source,
                "to": leecher,
                "piece": piece
            })
    
    # Execute all transfers simultaneously
    for transfer in planned_transfers:
        G.nodes[transfer["to"]]["file_pieces"].add(transfer["piece"])
        transfers.append(transfer)
    
    # Check for completions after all transfers
    for leecher in leechers:
        if update_agent_completion(G, leecher, total_pieces):
            new_completions.append(leecher)
    
    return {
        "transfers": transfers,
        "new_completions": new_completions,
        "total_transfers": len(transfers)
    }


def get_network_stats(G: nx.Graph, total_pieces: int) -> Dict:
    """
    Overview of the network
    """
    seeders = [node for node in G.nodes() if G.nodes[node].get("role") == "seeder"]
    leechers = [node for node in G.nodes() if G.nodes[node].get("role") == "leecher"]
    complete_leechers = [node for node in leechers if G.nodes[node].get("is_complete", False)]
    
    total_pieces_in_network = sum(len(G.nodes[node].get("file_pieces", set())) for node in G.nodes())
    
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "seeders": len(seeders),
        "leechers": len(leechers),
        "complete_leechers": len(complete_leechers),
        "incomplete_leechers": len(leechers) - len(complete_leechers),
        "total_pieces_in_network": total_pieces_in_network,
        "completion_rate": len(complete_leechers) / len(leechers) if leechers else 1.0
    }