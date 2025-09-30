import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

class AgentType(str, Enum):
    """Enumeration for different types of agents."""
    SEEDER = "seeder"
    LEECHER = "leecher"


@dataclass
class Agent:
    node_id: int
    agent_type: AgentType
    file_pieces: Set[int] = field(default_factory=set)
    upload_capacity: int = 1
    download_capacity: int = 1
    is_complete: bool = False


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


def update_agent_completion(G: nx.Graph, node_id: int, total_pieces: int) -> bool:
    """
    is complete is true if the agent has all file pieces
    """
    if node_id not in G.nodes():
        return False
    
    pieces = G.nodes[node_id].get("file_pieces", set())
    is_complete = len(pieces) >= total_pieces
    G.nodes[node_id]["is_complete"] = is_complete
    return is_complete