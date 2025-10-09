import random
from typing import Dict, List, Optional, Tuple

from src.agent import update_agent_completion, agent_gossip_behavior, process_gossip_message_at_node, get_network_stats
from utils.plotter import GraphPlotter

import networkx as nx



def simulate_round(G: nx.Graph, total_pieces: int, seed: Optional[int] = None, single_agent: Optional[int] = None, cleanup_completed_queries: bool = True, search_mode: str = "realistic", K: int = 3, ttl: int = 5, max_searches_per_round: int = 3, current_round: int = 0) -> Dict:
    """
    Main simulation function - step-by-step: queries, hits, transfers.
    
    Args:
        G: NetworkX graph
        total_pieces: Total number of file pieces
        seed: Random seed for reproducibility
        single_agent: Force only this agent to search (if specified)
        cleanup_completed_queries: Whether to clean up completed queries
        search_mode: Search initiation mode
        K: Number of neighbors to forward queries to
        ttl: Time-to-live for queries (number of hops)
        max_searches_per_round: Maximum concurrent searches (for limited mode)
        current_round: The current round number (for retry tracking)
    
    search_mode options:
    - "single": Only one (random) node searches per round
    - "realistic": Each agent decides independently
    """
    return simulate_step_by_step_round(G, total_pieces, K=K, ttl=ttl, max_searches_per_round=max_searches_per_round, seed=seed, single_agent=single_agent, cleanup_completed_queries=cleanup_completed_queries, search_mode=search_mode, current_round=current_round)

# Global vars between rounds to hold the currently pending
pending_queries = []
pending_hits = []

def clean_completed_queries(G: nx.Graph, completed_query_uuids: set) -> None:
    """Optionally clean up completed queries from pending lists and agent states from the simulation to make it look cleaner."""
    global pending_queries, pending_hits
    
    # Remove completed queries from pending
    pending_queries[:] = [q for q in pending_queries if q["query_uuid"] not in completed_query_uuids]
    
    # Remove completed hits from pending
    pending_hits[:] = [h for h in pending_hits if h["query_uuid"] not in completed_query_uuids]
    
    # Clean up agent
    for node in G.nodes():
        agent = G.nodes[node].get("agent_object")
        if agent:
            # Remove completed queries from seen_queries
            agent.seen_queries -= completed_query_uuids
            # Remove completed queries from routing
            for query_uuid in completed_query_uuids:
                agent.query_routing.pop(query_uuid, None)

def simulate_step_by_step_round(G: nx.Graph, total_pieces: int, K: int = 3, ttl: int = 5, max_searches_per_round: int = 3, seed: Optional[int] = None, single_agent: Optional[int] = None, cleanup_completed_queries: bool = True, search_mode: str = "realistic", current_round: int = 0) -> Dict:
    global pending_queries, pending_hits

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    all_transfers = []
    new_completions = []
    
    # Select an agent to initiate a search
    agent_actions = {}
    search_initiators = []
    
    if search_mode == "single":
        # Original single-node (One node starts a new search each round)
        leechers_needing_pieces = []
        for node in G.nodes():
            agent = G.nodes[node].get("agent_object")
            if not agent or agent.agent_type.value not in ["leecher", "hybrid"]:
                continue
            
            agent.file_pieces = G.nodes[node].get("file_pieces", set())
            needed_pieces = agent.get_needed_pieces(total_pieces)
            if needed_pieces:
                leechers_needing_pieces.append(node)
        
        if leechers_needing_pieces:
            if single_agent is not None and single_agent in leechers_needing_pieces:
                search_initiators = [single_agent]
            else:
                search_initiators = [rng.choice(leechers_needing_pieces)]
    
    elif search_mode == "realistic":
        # Each agent decides independently if they should search.
        for node in G.nodes():
            agent = G.nodes[node].get("agent_object")
            if not agent:
                agent_actions[node] = {"initiate_search": None, "respond_to_queries": [], "transfers": []}
                continue
            
            # Sync state
            agent.file_pieces = G.nodes[node].get("file_pieces", set())
            agent.is_complete = G.nodes[node].get("is_complete", False)
            
            # Let each agent decide
            if single_agent is not None and node != single_agent:
                agent_actions[node] = {"initiate_search": None, "respond_to_queries": [], "transfers": []}
            else:
                gossip_actions = agent.decide_gossip_actions(G, total_pieces, rng, K, ttl, current_round)
                agent_actions[node] = gossip_actions
                
                if gossip_actions["initiate_search"]:
                    search_initiators.append(node)
    
    # Process search initiators
    for node in search_initiators:
        agent = G.nodes[node].get("agent_object")
        needed_pieces = agent.get_needed_pieces(total_pieces)
        if needed_pieces:
            piece = rng.choice(list(needed_pieces))
            search_result = agent.initiate_gossip_query(piece, ttl, K, G, rng)
            
            if search_result["forwards"]:
                # Create initial queries (not just ONE)
                for target in search_result["forwards"]:
                    message = {
                        "type": "query",
                        "query_uuid": search_result["query_uuid"],
                        "piece": search_result["piece"],
                        "ttl": search_result["ttl"],
                        "K": K,
                        "from_node": node,
                        "to_node": target,
                        "origin": node
                    }
                    pending_queries.append(message)
            
            agent_actions[node] = {"initiate_search": search_result}
    
    # Set empty actions for all other non-searching nodes
    for node in G.nodes():
        if node not in search_initiators:
            if node not in agent_actions:
                agent_actions[node] = {"initiate_search": None, "respond_to_queries": [], "transfers": []}
    
    # 1. Process pending queries (one hop)
    current_queries = pending_queries.copy()
    pending_queries.clear()  # Clear pending queries
    hits_generated = []
    
    for query in current_queries:
        # Process query at target node pon arrival
        response = process_gossip_message_at_node(G, query["to_node"], query, seed)
        
        if response["type"] == "query_response":
            # Forward query to next nodes next round
            for target in response["forwards"]:
                next_query = {
                    "type": "query",
                    "query_uuid": query["query_uuid"],
                    "piece": query["piece"],
                    "ttl": query["ttl"] - 1,
                    "K": query["K"],
                    "from_node": query["to_node"],
                    "to_node": target,
                    "origin": query["origin"]
                }
                if next_query["ttl"] > 0:  # Only forward if TTL > 0
                    pending_queries.append(next_query)  # Add to pending for next round
            
            # If we found the piece, create hit message to send back
            if response["hit"] is not None:
                hit_message = {
                    "type": "hit",
                    "query_uuid": query["query_uuid"],
                    "piece": query["piece"],
                    "from_node": query["to_node"],
                    "to_node": query["from_node"],  # Send back to immediate sender (reverse path)
                    "hit_node": response["hit"],
                    "origin": query["origin"]
                }
                hits_generated.append(hit_message)
    
    #2. Process pending hits (one hop back next round)
    current_hits = pending_hits.copy() + hits_generated
    pending_hits.clear()
    
    for hit in current_hits:
        # Reached the origin?
        if hit["to_node"] == hit["origin"]:
            # No need to process further, just mark for transfer
            continue
        
        # Handle the Gossip message.
        response = process_gossip_message_at_node(G, hit["to_node"], hit, seed)
        
        if response["type"] == "hit_forward":
            # Forward hit back to next forward node nex round
            for target in response["forwards"]:
                next_hit = {
                    "type": "hit",
                    "query_uuid": hit["query_uuid"],
                    "piece": hit["piece"],
                    "from_node": hit["to_node"],
                    "to_node": target,
                    "hit_node": hit.get("hit_node", response.get("hit_node")),
                    "origin": hit["origin"]  # Always use the original origin.
                }
                pending_hits.append(next_hit)  # Add to pending for next round
    
    # 3. Create transfers, preventing duplicates for the same piece to the same node
    processed_transfers = set()
    
    for hit in current_hits:
        if hit["to_node"] == hit["origin"]:  # Hit reached origin
            # Did we already have a transfer?
            transfer_key = (hit["to_node"], hit["piece"])
            if transfer_key in processed_transfers:
                continue  # Skip
            
            # Check if the origin already has this piece
            origin_pieces = G.nodes[hit["to_node"]].get("file_pieces", set())
            if hit["piece"] in origin_pieces:
                continue  # Skip
            
            # Create transfer
            transfer = {
                "from": hit["hit_node"],
                "to": hit["to_node"],
                "piece": hit["piece"],
                "query_uuid": hit["query_uuid"]
            }
            all_transfers.append(transfer)
            processed_transfers.add(transfer_key)
            
            # Grant the piece (simulate the transfer)
            G.nodes[transfer["to"]].setdefault("file_pieces", set()).add(transfer["piece"])
            
            # Check for completion
            if update_agent_completion(G, transfer["to"], total_pieces):
                new_completions.append(transfer["to"])
            
            # Clean up search tracking for the piece that was just found
            agent = G.nodes[transfer["to"]].get("agent_object")
            if agent:
                agent.clear_found_pieces()
    
    # Clean up completed queries if enabled for cleaner graph
    if cleanup_completed_queries and all_transfers:
        completed_query_uuids = {transfer["query_uuid"] for transfer in all_transfers}
        clean_completed_queries(G, completed_query_uuids)
    
    # debug
    for query in current_queries:
        query_path = [query["origin"]]
        if query["from_node"] != query["origin"]:
            query_path.append(query["from_node"])
        query_path.append(query["to_node"])
        
        query["debug_info"] = {
            "query_uuid": query["query_uuid"],
            "piece": query["piece"],
            "origin": query["origin"],
            "from_node": query["from_node"],
            "to_node": query["to_node"],
            "ttl": query["ttl"],
            "path": query_path
        }
    
    for hit in current_hits:
        # reverse path
        hit_path = [hit["hit_node"]]
        if hit["from_node"] != hit["hit_node"]:
            hit_path.append(hit["from_node"])
        hit_path.append(hit["to_node"])
        
        hit["debug_info"] = {
            "query_uuid": hit["query_uuid"],
            "piece": hit["piece"],
            "origin": hit["origin"],
            "hit_node": hit["hit_node"],
            "current_node": hit["from_node"],
            "target_node": hit["to_node"],
            "path": hit_path
        }
    
    message_rounds = [
        current_queries,  # Queries
        current_hits,     # Hits
        []                # Transfers (no messages, just transfers)
    ]
    
    # Clean up old info
    # Collect all active query UUIDs from current round AND pending messages
    active_query_uuids = set()
    for query in current_queries:
        active_query_uuids.add(query["query_uuid"])
    for hit in current_hits:
        active_query_uuids.add(hit["query_uuid"])
    for query in pending_queries:
        active_query_uuids.add(query["query_uuid"])
    for hit in pending_hits:
        active_query_uuids.add(hit["query_uuid"])
    
    
    for node in G.nodes():
        agent = G.nodes[node].get("agent_object")
        if agent:
            # Clear search tracking for pieces that have been found
            agent.clear_found_pieces()
            # Only clear completed queries, keep active ones
            agent.clear_completed_queries(active_query_uuids)
            # Still do the old cleanup
            agent.clear_old_queries()
    
    return {
        "messages": current_queries + current_hits, # All messages this round
        "transfers": all_transfers, # All transfers this round
        "new_completions": new_completions, # Agents who completed this round
        "total_messages": len(current_queries + current_hits), # Total messages this round
        "total_transfers": len(all_transfers), # Total transfers this round
        "message_rounds": message_rounds, # Messages grouped by type
        "agent_actions": agent_actions, # Actions taken by each agent this round
        "search_initiators": search_initiators, # All agents who initiated searches this round
        "num_searchers": len(search_initiators) # Number of agents that searched this round
    }

