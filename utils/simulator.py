import random
from typing import Dict, Optional

from src.agent import update_agent_completion, agent_behavior

import networkx as nx


def simulate_round(G: nx.Graph, total_pieces: int, seed: Optional[int] = None) -> Dict:
    """
    Simo sending pieces to leechers
    """
    rng = random.Random(seed)
    transfers = []
    new_completions = []
    
    #(agents who need pieces)
    all_actions = {}
    for node in G.nodes():
        node_seed = rng.randint(1, 10000) if seed is not None else None
        all_actions[node] = agent_behavior(G, node, total_pieces, node_seed)
    
    # 1. Collect all requests
    all_requests = []
    for node, actions in all_actions.items():
        for source, piece in actions["requests"]:
            all_requests.append({
                "requester": node,
                "source": source,
                "piece": piece
            })
    # 2. Respond to requests from any Node
    node_responses = {}
    for node, actions in all_actions.items():
        node_pieces = G.nodes[node].get("file_pieces", set())
        if node_pieces:
            node_responses[node] = []
            uploads_planned = 0
            upload_capacity = G.nodes[node].get("agent_object").upload_capacity if G.nodes[node].get("agent_object") else 1
            
            requests_to_this_node = [req for req in all_requests if req["source"] == node]
            rng.shuffle(requests_to_this_node)
            
            for request in requests_to_this_node:
                if uploads_planned >= upload_capacity:
                    break
                
                # Check if node has the requested piece
                if request["piece"] in node_pieces:
                    node_responses[node].append(request)
                    uploads_planned += 1
    
    # 3. Execute transfers
    for node, responses in node_responses.items():
        for response in responses:
            requester = response["requester"]
            piece = response["piece"]
            
            # Add piece to requester
            G.nodes[requester]["file_pieces"].add(piece)
            transfers.append({
                "from": node,
                "to": requester,
                "piece": piece
            })
    
    # Check for completions after all transfers
    for node in G.nodes():
        if update_agent_completion(G, node, total_pieces):
            new_completions.append(node)
    
    return {
        "transfers": transfers,
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