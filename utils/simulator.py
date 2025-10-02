import random
from typing import Dict, Optional

from src.agent import update_agent_completion, agent_behavior, agent_upload_behavior
from utils.plotter import GraphPlotter

import networkx as nx


def simulate_round(G: nx.Graph, total_pieces: int, seed: Optional[int] = None) -> Dict:
    """
    Simo sending pieces to leechers
    """
    rng = random.Random(seed)
    transfers = []
    new_completions = []
    
    #(agents who need pieces)
    all_requests = []
    for node in G.nodes():
        node_seed = rng.randint(1, 10000) if seed is not None else None
        actions = agent_behavior(G, node, total_pieces, node_seed)
        
        # 1. Collect all requests
        for source, piece in actions["requests"]:
            all_requests.append({
                "requester": node,
                "source": source,
                "piece": piece
            })
    
    # 2. Sort the requests from any Node
    requests_by_target = {}
    for request in all_requests:
        target = request["source"]
        if target not in requests_by_target:
            requests_by_target[target] = []
        requests_by_target[target].append((request["requester"], request["piece"]))
    
    # 3. Decide on uploads
    for target_node, incoming_requests in requests_by_target.items():
        target_seed = rng.randint(1, 10000) if seed is not None else None
        uploads = agent_upload_behavior(G, target_node, incoming_requests, target_seed)
        
        # 4. Execute uploads
        for requester, piece in uploads:
            # Add piece to requester
            G.nodes[requester]["file_pieces"].add(piece)
            transfers.append({
                "from": target_node,
                "to": requester,
                "piece": piece
            })
    
    # 5. Check for completed nodes
    for node in G.nodes():
        if update_agent_completion(G, node, total_pieces):
            new_completions.append(node)
    
    return {
        "transfers": transfers,
        "requests": all_requests,
        "new_completions": new_completions,
        "total_transfers": len(transfers),
        "total_requests": len(all_requests),
        "fulfilled_requests": len(transfers)
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