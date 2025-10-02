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


def simulate_with_saved_images(G: nx.Graph, total_pieces: int, rounds: int = 5, 
                              seed: Optional[int] = None) -> str:
    """Run simulation and save images of each round to a timestamped folder."""
    print(f"🚀 Starting P2P simulation with image saving: {G.number_of_nodes()} nodes, {rounds} rounds")
    
    # Create plotter with image saving enabled
    plotter = GraphPlotter(save_images=True)
    print(f"📁 Saving images to: {plotter.output_dir}")
    
    # Show initial state
    plotter.draw_base_graph(G, total_pieces, "Initial Network State")
    
    # Run simulation rounds
    for round_num in range(1, rounds + 1):
        result = simulate_round(G, total_pieces, seed)
        
        print(f"\n📊 Round {round_num}:")
        print(f"   Requests: {result['total_requests']}")
        print(f"   Fulfilled: {result['fulfilled_requests']}")
        print(f"   Transfers: {result['total_transfers']}")
        
        if result['transfers']:
            print("   Transfers:")
            for transfer in result['transfers']:
                print(f"     Piece {transfer['piece']}: {transfer['from']} → {transfer['to']}")
        
        if result['new_completions']:
            print(f"   🎉 New completions: {result['new_completions']}")
        
        # Show visualization with transfers and save image
        plotter.draw_with_transfers(G, total_pieces, result['transfers'], round_num)
    
    # Show final stats
    final_stats = get_network_stats(G, total_pieces)
    print(f"\n🏁 Final Results:")
    print(f"   Completion rate: {final_stats['completion_rate']*100:.1f}%")
    print(f"   Complete leechers: {final_stats['complete_leechers']}/{final_stats['leechers']}")
    print(f"   Total pieces in network: {final_stats['total_pieces_in_network']}")
    
    return plotter.output_dir