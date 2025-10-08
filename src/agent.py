import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

class AgentType(str, Enum):
    """Enumeration for different types of agents."""
    SEEDER = "seeder"
    LEECHER = "leecher"
    HYBRID = "hybrid"


@dataclass
class Agent:
    node_id: int
    agent_type: AgentType
    file_pieces: Set[int] = field(default_factory=set)
    upload_capacity: int = 1
    download_capacity: int = 1
    is_complete: bool = False
    seen_queries: Set[str] = field(default_factory=set)  # UUIDs of queries processed (for duplication check)
    query_routing: Dict[str, int] = field(default_factory=dict)  # query_uuid -> who_forwarded_it_to_me
    
    def get_needed_pieces(self, total_pieces: int) -> Set[int]:
        """Get pieces this agent needs to complete the file."""
        return set(range(total_pieces)) - self.file_pieces
    
    def get_available_pieces_for_upload(self, requester_pieces: Set[int]) -> Set[int]:
        """Get pieces this agent can upload to a specific requester."""
        return self.file_pieces - requester_pieces
    
    def can_upload(self, current_uploads: int) -> bool:
        """Check if this agent can perform another upload."""
        return current_uploads < self.upload_capacity
    
    def can_download(self, current_downloads: int) -> bool:
        """Check if this agent can perform another download."""
        return current_downloads < self.download_capacity
    
    def decide_gossip_actions(self, graph: nx.Graph, total_pieces: int, rng: random.Random, K: int = 3, ttl: int = 5) -> Dict:
        """Decide whether to initiate gossip searches or respond to queries.
        This replaces the old request/transfer logic with gossip discovery.
         Returns a dict with actions taken.
         """
        actions = {
            "initiate_search": None,
            "respond_to_queries": [],
            "transfers": []
        }
        
        self.file_pieces = graph.nodes[self.node_id].get("file_pieces", set())
        
        # If we're a leecher or hybrid and need pieces, decide whether to search
        # Barebones, 30% chance to initiate a search if we need pieces
        # initiate a query up to K neighbors with TTL
        if self.agent_type in [AgentType.LEECHER, AgentType.HYBRID]:
            needed_pieces = self.get_needed_pieces(total_pieces)
            if needed_pieces and rng.random() < 0.3:
                piece = rng.choice(list(needed_pieces))
                search_result = self.initiate_gossip_query(piece, ttl, K, graph, rng)
                actions["initiate_search"] = search_result
        
        return actions
    
    def get_link_quality(self, graph: nx.Graph, neighbor: int) -> float:
        """Get the bandwidth between this node and neighbor."""
        if graph.has_edge(self.node_id, neighbor):
            return graph[self.node_id][neighbor].get('weight', 50.0)  # Default to 50 if no weight
        return 0.0  # No direct connection
    
    def process_gossip_message(self, message: Dict, graph: nx.Graph, rng: random.Random) -> Dict:
        """Process incoming gossip messages (queries or hits) and return response based on type."""
        response = {
            "type": "none",
            "forwards": [],
            "hit": None,
            "transfer": None
        }
        
        if message["type"] == "query":
            # Process incoming if type is query
            result = self.process_gossip_query(
                message["query_uuid"], 
                message["piece"], 
                message["ttl"], 
                message["from_node"], 
                message["K"], 
                graph, 
                rng
            )
            
            response["type"] = "query_response"
            response["forwards"] = result["forwards"]
            response["hit"] = result["hit"]
            
            # "If we have the piece", create a transfer (but check for duplicates)
            if result["hit"] is not None:
                # Check if the origin already has this piece to prevent unnecessary transfers
                origin_pieces = graph.nodes[message["origin"]].get("file_pieces", set())
                if message["piece"] not in origin_pieces:
                    response["transfer"] = {
                        "from": self.node_id,
                        "to": message["origin"],
                        "piece": message["piece"],
                        "query_uuid": message["query_uuid"]
                    }
        
        elif message["type"] == "hit":
            # Process QueryHit response
            hit_result = self.process_gossip_hit(
                message["query_uuid"], 
                message["hit_node"], 
                graph
            )
            response.update(hit_result)
        return response
    
    def initiate_gossip_query(self, piece: int, ttl: int, K: int, graph: nx.Graph, rng: random.Random) -> Dict:
        """Initiate a gossip query for a specific piece."""
        query_uuid = str(uuid.uuid4()) # random
        self.seen_queries.add(query_uuid) # Mark as seen to avoid re-proc our own query
        
        # Choose K neighbors to forward to, biased by edge link quality (bandwidth) weights
        neighbors = list(graph.neighbors(self.node_id))
        if not neighbors:
            return {"query_uuid": query_uuid, "forwards": [], "hit": None}
        
        # Weight neighbors
        weighted_neighbors = []
        for neighbor in neighbors:
            link_quality = self.get_link_quality(graph, neighbor)
            weighted_neighbors.append((neighbor, link_quality))
        
        # Sort by link quality - prefer better connections
        weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
        selected_neighbors = [n for n, _ in weighted_neighbors[:K]]
        
        return {
            "query_uuid": query_uuid,
            "piece": piece,
            "ttl": ttl,
            "forwards": selected_neighbors,
            "hit": None
        }
    
    def process_gossip_query(self, query_uuid: str, piece: int, ttl: int, from_node: int, K: int, graph: nx.Graph, rng: random.Random) -> Dict:
        """Process an incoming gossip query."""
        # Have we seen this query before? If so, ignore it and mark as duplicate
        if query_uuid in self.seen_queries:
            return {"query_uuid": query_uuid, "forwards": [], "hit": None, "duplicate": True}
        
        # Record this query and who forwarded it
        self.seen_queries.add(query_uuid)
        self.query_routing[query_uuid] = from_node
        
        # Sync agent state with graph state
        self.file_pieces = graph.nodes[self.node_id].get("file_pieces", set())
        
        # Check if we have the piece (seeders, hybrids, and leechers with pieces can respond)
        hit = None
        if piece in self.file_pieces:
            hit = self.node_id
        
        # Forward to K neighbors if TTL > 0
        forwards = []
        if ttl > 0:
            neighbors = list(graph.neighbors(self.node_id))
            # Exclude the node that forwarded to us and already seen nodes
            candidates = [n for n in neighbors if n != from_node]
            
                 # Weight neighbors
            if candidates:
                weighted_candidates = []
                for neighbor in candidates:
                    link_quality = self.get_link_quality(graph, neighbor)
                    weighted_candidates.append((neighbor, link_quality))
                
                # Sort by link quality - prefer better connections
                weighted_candidates.sort(key=lambda x: x[1], reverse=True)
                forwards = [n for n, _ in weighted_candidates[:K]]
        
        return {
            "query_uuid": query_uuid,
            "piece": piece,
            "ttl": ttl - 1,
            "forwards": forwards,
            "hit": hit,
            "duplicate": False
        }
    
    def process_gossip_hit(self, query_uuid: str, hit_node: int, graph: nx.Graph) -> Dict:
        """Process a QueryHit response."""
        if query_uuid not in self.query_routing:
            return {"type": "hit_forward", "forwards": [], "hit": None, "transfer": None}
        
        # Route back to whoever forwarded us this query
        next_hop = self.query_routing[query_uuid]
        return {
            "type": "hit_forward",
            "forwards": [next_hop],
            "hit": hit_node,
            "transfer": None
        }
    
    def clear_old_queries(self, max_age: int = 100) -> None:
        """Clear old queries that are too old to exist"""
        if len(self.seen_queries) > max_age:
            self.seen_queries.clear()
            # Don't clear query_routing here as it's needed for hit routing
            # The clear_completed_queries function handles routing cleanup
    
    def clear_completed_queries(self, active_query_uuids: set) -> None:
        """Clear routing for queries that are no longer active."""
        completed_queries = set(self.query_routing.keys()) - active_query_uuids
        for query_uuid in completed_queries:
            self.query_routing.pop(query_uuid, None)


# DEPRECATED
def normal_prob(probs: Dict[AgentType, float]) -> Dict[AgentType, float]:
    total = float(sum(probs.values()))
    if total <= 0:
        raise ValueError("Sum of probabilities must be greater than zero.")
    return {k: v / total for k, v in probs.items()}

def assign_random_agent_types(G: nx.Graph, probs: Optional[Dict[AgentType, float]] = None, seed: Optional[int] = None) -> Dict[int, Agent]:
    if probs is None:
        # Default test
        probs = {
            AgentType.SEEDER: 0.25,
            AgentType.LEECHER: 0.25
        }

    probs = normal_prob(probs)
    rng = random.Random(seed)

    roles = list(probs.keys())
    weights = [probs[role] for role in roles]

    agents: Dict[int, Agent] = {}
    for node in G.nodes:
        agent_type = rng.choices(roles, weights=weights)[0]
        agents[node] = Agent(node_id=node, agent_type=agent_type)
        G.nodes[node]["role"] = agent_type.value
        G.nodes[node]["agent_object"] = agents[node]

    return agents

def assign_n_seeders(G: nx.Graph, n: int, seed: Optional[int] = None) -> Dict[int, Agent]:
    rng = random.Random(seed)
    nodes = list(G.nodes)
    if n > len(nodes):
        raise ValueError("n cannot be greater than the number of nodes in the graph.")

    seeders = set(rng.sample(nodes, n))
    agents: Dict[int, Agent] = {}
    for node in nodes:
        if node in seeders:
            agent_type = AgentType.SEEDER
        else:
            agent_type = AgentType.LEECHER
        agents[node] = Agent(node_id=node, agent_type=agent_type)
        G.nodes[node]["role"] = agent_type.value
        G.nodes[node]["agent_object"] = agents[node]

    return agents


def add_node(G: nx.Graph, agent_type: AgentType, node_id: Optional[int] = None, 
             connect_to_existing: bool = True, connection_prob: float = 0.3) -> Tuple[int, Agent]:
    """
    Add a new node to the graph with the specified agent type.
    """

    if node_id is None:
        # Find the next available node ID
        existing_ids = set(G.nodes())
        node_id = 0
        while node_id in existing_ids:
            node_id += 1
    
    if node_id in G.nodes():
        raise ValueError(f"Node {node_id} already exists in the graph.")
    
    # Add the node to the graph
    G.add_node(node_id)
    agent = Agent(node_id=node_id, agent_type=agent_type)
    G.nodes[node_id]["role"] = agent_type.value
    G.nodes[node_id]["agent_object"] = agent
    
    # Connect to existing nodes if requested
    if connect_to_existing and len(G.nodes()) > 1:
        existing_nodes = [n for n in G.nodes() if n != node_id]
        for existing_node in existing_nodes:
            if random.random() < connection_prob:
                G.add_edge(node_id, existing_node)
    
    return node_id, agent


def remove_node(G: nx.Graph, node_id: int) -> bool:
    """
    Remove a node from the graph.
    """
    if node_id not in G.nodes():
        return False
    
    G.remove_node(node_id)
    return True


def add_random_node(G: nx.Graph, agent_type_probs: Optional[Dict[AgentType, float]] = None,
                   node_id: Optional[int] = None, connect_to_existing: bool = True, 
                   connection_prob: float = 0.3, seed: Optional[int] = None) -> Tuple[int, Agent]:
    """
    Add a new node to the graph with a randomly assigned agent type.
    """
    if agent_type_probs is None:
        agent_type_probs = {
            AgentType.SEEDER: 0.5,
            AgentType.LEECHER: 0.5
        }
    
    agent_type_probs = normal_prob(agent_type_probs)
    rng = random.Random(seed)
    
    agent_types = list(agent_type_probs.keys())
    weights = [agent_type_probs[agent_type] for agent_type in agent_types]
    chosen_agent_type = rng.choices(agent_types, weights=weights)[0]
    
    return add_node(G, chosen_agent_type, node_id, connect_to_existing, connection_prob)


def initialize_file_sharing(G: nx.Graph, file_size_pieces: int, seed: Optional[int] = None) -> None:
    """
    File sharing, give seeders all file pieces and leechers none as a test
    """
    rng = random.Random(seed)
    
    for node, data in G.nodes(data=True):
        if "role" in data:
            agent_type = AgentType(data["role"])
            if agent_type == AgentType.SEEDER:
                # Seeders start with all file pieces
                G.nodes[node]["file_pieces"] = set(range(file_size_pieces))
                G.nodes[node]["is_complete"] = True
            else:  # LEECHER (just a placeholder for now)
                # Leechers start with no pieces
                G.nodes[node]["file_pieces"] = set()
                G.nodes[node]["is_complete"] = False


def get_agent_info(G: nx.Graph, node_id: int) -> Optional[Dict]:
    """
    Get agent information
    """
    if node_id not in G.nodes():
        return None
    
    data = G.nodes[node_id]
    return {
        "node_id": node_id,
        "role": data.get("role", "unknown"),
        "file_pieces": data.get("file_pieces", set()),
        "num_pieces": len(data.get("file_pieces", set())),
        "is_complete": data.get("is_complete", False),
        "neighbors": list(G.neighbors(node_id))
    }


def change_agent_role(G: nx.Graph, node_id: int, new_role: str) -> bool:
    """
    Update an Agent Role
    """
    if node_id not in G.nodes():
        return False

    if new_role not in ["seeder", "leecher", "hybrid"]:
        raise ValueError(f"Invalid role: {new_role}. Must be 'seeder', 'leecher', or 'hybrid'")
    
    G.nodes[node_id]["role"] = new_role
    
    agent_obj = G.nodes[node_id].get("agent_object")
    if agent_obj:
        agent_obj.agent_type = AgentType(new_role)
    
    return True


def update_agent_completion(G: nx.Graph, node_id: int, total_pieces: int) -> bool:
    """
    is complete is true if the agent has all file pieces
    """
    if node_id not in G.nodes():
        return False
    
    pieces = G.nodes[node_id].get("file_pieces", set())
    was_complete = G.nodes[node_id].get("is_complete", False)
    is_complete = len(pieces) >= total_pieces
    
    G.nodes[node_id]["is_complete"] = is_complete
    
    current_role = G.nodes[node_id].get("role")
    agent_obj = G.nodes[node_id].get("agent_object")
    
    if is_complete and not was_complete:
        # Agent just became complete - convert to seeder
        if current_role in ["leecher", "hybrid"]:
            change_agent_role(G, node_id, "seeder")
            if agent_obj:
                agent_obj.is_complete = True
    elif not is_complete and len(pieces) > 0 and current_role == "leecher":
        # Agent has some pieces but not all - convert to hybrid
        change_agent_role(G, node_id, "hybrid")
        if agent_obj:
            agent_obj.agent_type = AgentType.HYBRID
    
    # true only if agent just became complete (was incomplete, now complete)
    return is_complete and not was_complete



## GOSSIP-BASED AGENT FUNCTIONS

def agent_gossip_behavior(G: nx.Graph, node_id: int, total_pieces: int, K: int = 3, ttl: int = 5, 
                          seed: Optional[int] = None) -> Dict:
    """
    Agent Query/QueryHit Behavior using Gossip.
    """
    rng = random.Random(seed)
    actions = {
        "initiate_search": None, # if initiated, details of the search
        "messages": [], # messages sent (queries/hits)
        "transfers": [] # transfers made
    }
    
    if node_id not in G.nodes():
        return actions
    
    agent_data = G.nodes[node_id]
    agent_obj = agent_data.get("agent_object")
    
    if not agent_obj:
        return actions
    
    agent_obj.file_pieces = agent_data.get("file_pieces", set())
    agent_obj.is_complete = agent_data.get("is_complete", False)
    
    # Only complete seeders don't send queries anymore
    if agent_obj.is_complete and agent_obj.agent_type == AgentType.SEEDER:
        return actions
    
    # Decide gossip actions
    gossip_actions = agent_obj.decide_gossip_actions(G, total_pieces, rng, K, ttl)
    actions.update(gossip_actions)
    
    return actions


def process_gossip_message_at_node(G: nx.Graph, node_id: int, message: Dict, 
                                  seed: Optional[int] = None) -> Dict:
    """
    Process a gossip message at a specific node.
    """
    rng = random.Random(seed)
    
    if node_id not in G.nodes():
        return {"type": "none", "forwards": [], "hit": None, "transfer": None}
    
    agent_obj = G.nodes[node_id].get("agent_object")
    if not agent_obj:
        return {"type": "none", "forwards": [], "hit": None, "transfer": None}
    
    # Sync agent state
    agent_obj.file_pieces = G.nodes[node_id].get("file_pieces", set())
    
    return agent_obj.process_gossip_message(message, G, rng)


def get_network_stats(G: nx.Graph, total_pieces: int) -> Dict:
    """
    Overview of the network
    """
    seeders = [node for node in G.nodes() if G.nodes[node].get("role") == "seeder"]
    leechers = [node for node in G.nodes() if G.nodes[node].get("role") == "leecher"]
    hybrids = [node for node in G.nodes() if G.nodes[node].get("role") == "hybrid"]
    complete_leechers = [node for node in leechers if G.nodes[node].get("is_complete", False)]
    complete_hybrids = [node for node in hybrids if G.nodes[node].get("is_complete", False)]
    
    total_pieces_in_network = sum(len(G.nodes[node].get("file_pieces", set())) for node in G.nodes())
    
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "seeders": len(seeders),
        "leechers": len(leechers),
        "hybrids": len(hybrids),
        "complete_leechers": len(complete_leechers),
        "complete_hybrids": len(complete_hybrids),
        "incomplete_leechers": len(leechers) - len(complete_leechers),
        "incomplete_hybrids": len(hybrids) - len(complete_hybrids),
        "total_pieces_in_network": total_pieces_in_network,
        "completion_rate": (len(complete_leechers) + len(complete_hybrids)) / (len(leechers) + len(hybrids)) if (leechers or hybrids) else 1.0
    }