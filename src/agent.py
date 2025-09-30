import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx

class AgentType(str, Enum):
    """Enumeration for different types of agents."""
    SEEDER = "seeder"
    LEECHER = "leecher"
    FREE_RIDER = "free_rider"
    ALTRUIST = "altruist"

@dataclass
class Agent:
    node_id: int
    agent_type: AgentType


def normal_prob(probs: Dict[AgentType, float]) -> Dict[AgentType, float]:
    total = float(sum(probs.values()))
    if total <= 0:
        raise ValueError("Sum of probabilities must be greater than zero.")
    return {k: v / total for k, v in probs.items()}

def assign_agent_types(G: nx.Graph, probs: Optional[Dict[AgentType, float]] = None, seed: Optional[int] = None) -> Dict[int, Agent]:
    if probs is None:
        # Default test
        probs = {
            AgentType.SEEDER: 0.25,
            AgentType.LEECHER: 0.25,
            AgentType.FREE_RIDER: 0.25,
            AgentType.ALTRUIST: 0.25,
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