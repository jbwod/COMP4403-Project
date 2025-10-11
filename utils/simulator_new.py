import random
from typing import Dict, List, Optional, Tuple

from src.agent import get_network_stats, reset_agent_state
from utils.plotter import GraphPlotter

import networkx as nx


########################################################
# Environment Class
########################################################

class Environment:
    """Environment handles message delivery between agents and time-tick."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
    
    def deliver(self, current_round: int = 0, total_pieces: int = 15) -> Tuple[List[Dict], List[Dict]]:
        """Deliver all outbox messages to target agent inboxes. Returns (transfers, completions)."""
        transfers = []
        completions = []
        
        all_messages = []
        for node in self.graph.nodes():
            agent = self.graph.nodes[node].get('agent_object')
            if agent:
                all_messages.extend(agent.outbox)
                agent.outbox.clear()  # Clear
        
        # Deliver messages
        for message in all_messages:
            target_node = message["to_node"]
            if target_node in self.graph.nodes():
                target_agent = self.graph.nodes[target_node].get('agent_object')
                if target_agent:
                    target_agent.inbox.append(message)

        return transfers, completions


########################################################
# Main Simulation Function
########################################################

def simulate_round_agent_driven(G: nx.Graph, total_pieces: int, seed: Optional[int] = None, 
                                single_agent: Optional[int] = None, cleanup_completed_queries: bool = True, 
                                search_mode: str = "realistic", K: int = 3, ttl: int = 5, 
                                max_searches_per_round: int = 3, current_round: int = 0, 
                                neighbor_selection: str = "bandwidth") -> Dict:
    """
    Main simulation function - all message handling migrated to agent.py
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Create environment
    env = Environment(G)
    
    # Collect messages
    all_messages = []
    for node in G.nodes():
        agent = G.nodes[node].get("agent_object")
        if agent:
            all_messages.extend(agent.inbox)  # Messages that will be processed
    
    # Let each agent take one 'tick'
    search_initiators = []
    for node in G.nodes():
        agent = G.nodes[node].get("agent_object")
        if not agent:
            continue
        
        agent.tick(G, total_pieces, rng, K, ttl, current_round, neighbor_selection, single_agent)
        
        # Track search initiators (agents that created query messages)
        # But only count as search initiator if single_agent allows it
        if any(msg["type"] == "query" for msg in agent.outbox):
            if single_agent is None or node == single_agent:
                search_initiators.append(node)
    
    # Process transfer action
    transfers = []
    completions = []
    for node in G.nodes():
        agent = G.nodes[node].get("agent_object")
        if agent:
            # Process transfer actions in outbox
            transfer_actions = [msg for msg in agent.outbox if msg["type"] == "transfer"]
            # print(f"Node {node} has {len(transfer_actions)} transfer actions")
            for transfer_action in transfer_actions:
                piece = transfer_action["piece"]
                hit_node = transfer_action["from"]
                
                # print(f"Creating a piece transfer {hit_node} -> {node}, piece {piece}")
                # print(f"Agent {node} currently has {len(agent.file_pieces)} pieces")
                
                # Check if we don't already have this piece
                if piece not in agent.file_pieces:
                    # Execute
                    agent.on_piece_received(piece, total_pieces)
                    
                    # Update
                    G.nodes[node].setdefault("file_pieces", set()).add(piece)
                    
                    # Record
                    transfers.append({
                        "from": hit_node,
                        "to": node,
                        "piece": piece,
                        "query_uuid": transfer_action["query_uuid"]
                    })
                    
                    if agent.is_complete:
                        completions.append(node)
                else:
                    # print(f"Agent {node} already has piece {piece} so ignore")
                    pass
            
            # Remove transfer actions from outbox
            agent.outbox = [msg for msg in agent.outbox if msg["type"] != "transfer"]
    
    # Deliver messages
    env.deliver(current_round, total_pieces)
    
    # Update graph state to match agent state
    for node in G.nodes():
        agent = G.nodes[node].get("agent_object")
        if agent:
            G.nodes[node]["role"] = agent.agent_type.value
            G.nodes[node]["is_complete"] = agent.is_complete
    
    # Separate queries and hits for the graph drawing
    queries = [msg for msg in all_messages if msg["type"] == "query"]
    hits = [msg for msg in all_messages if msg["type"] == "hit"]
    
    return {
        "messages": all_messages,
        "transfers": transfers,
        "new_completions": completions,
        "total_messages": len(all_messages),
        "total_transfers": len(transfers),
        "message_rounds": [queries, hits, []],
        "agent_actions": {},  # not needed
        "search_initiators": search_initiators,
        "num_searchers": len(search_initiators)
    }


########################################################
# Reset Simulation
########################################################

def reset_simulation(G: nx.Graph, file_size_pieces: int, seed: Optional[int] = None) -> None:
    """
    1. Resets all agent states (file pieces, query tracking, etc.)
    2. Resets graph node attributes
    3. Clears agent mailboxes
    """
    for node in G.nodes():
        agent = G.nodes[node].get("agent_object")
        if agent:
            reset_agent_state(agent)

        G.nodes[node]["file_pieces"] = set()
        G.nodes[node]["is_complete"] = False