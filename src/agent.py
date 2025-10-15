import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

########################################################
# Roles
########################################################

class AgentType(str, Enum):
    """Enumeration for different types of agents."""
    SEEDER  =    "seeder"
    LEECHER =   "leecher"
    HYBRID  =    "hybrid"
    DEAD    =    "dead"


########################################################
# Primary Agent Class and Helper Functions

# get_needed_pieces
# get_retry_pieces

# decide_gossip_actions

# process_gossip_message
# process_gossip_query
# process_gossip_hit

# initiate_gossip_query
# get_link_quality
# select_neighbors

# clear_old_queries
# clear_completed_queries
# clear_found_pieces
# mark_failed_piece
########################################################
@dataclass
class Agent:
    node_id:        int
    agent_type:     AgentType
    file_pieces:    Set[int] = field(default_factory=set)

    is_complete:        bool = False
    seen_queries:       Set[str] = field(default_factory=set)  # UUIDs of queries processed (for duplication check)
    query_routing:      Dict[str, int] = field(default_factory=dict)  # query_uuid -> who_forwarded_it_to_me
    failed_pieces:      Dict[int, int] = field(default_factory=dict)  # piece -> retry_count
    last_search_time:   Dict[int, int] = field(default_factory=dict)  # piece -> round_when_last_searched
    failed_piece_time:  Dict[int, int] = field(default_factory=dict)  # piece -> round_when_failed
    
    # Per-agent in.out messages
    inbox:          List[Dict] = field(default_factory=list)  # Messages received this round
    outbox:         List[Dict] = field(default_factory=list)  # Messages to send next round
    
    # DEAD state management
    stored_edges:   List[Tuple[int, int, Dict]] = field(default_factory=list)  # (neighbor, weight, attributes) for killed nodes
    original_role:  Optional[AgentType] = None  # Original role before being killed
    
    ########################################################
    # Main Agent Tick Method
    ########################################################
    
    def tick(self, graph: nx.Graph, total_pieces: int, rng: random.Random, K: int = 3, ttl: int = 5, current_round: int = 0, neighbor_selection: str = "bandwidth", single_agent: Optional[int] = None) -> None:
        """Main agent tick - process inbox, decide actions, create outbox messages."""
        # DEAD agents don't DO anything
        if self.agent_type == AgentType.DEAD:
            return
            
        self.file_pieces = graph.nodes[self.node_id].get("file_pieces", set())
        
        # Process all messages in an agent inbox
        for message in self.inbox:
            self.process_gossip_message(message, graph, rng, neighbor_selection)
        
        # Clear inbox after we are done
        self.inbox.clear()
        
        # Decide gossip actions
        self.decide_gossip_actions(graph, total_pieces, rng, K, ttl, current_round, neighbor_selection, single_agent)
        
        # Handle transfers TODO; Not sure if we just leave this as simulator based.
        self.process_transfers(graph, total_pieces)
    
    def process_transfers(self, graph: nx.Graph, total_pieces: int) -> None:
        """Process any transfers that occurred this round."""
        # Handled this in the simulator
        pass
    
    def on_piece_received(self, piece: int, total_pieces: int) -> bool:
        """Updated on piece - True if agent just completed."""
        self.file_pieces.add(piece)
        
        was_complete = self.is_complete
        self.is_complete = len(self.file_pieces) >= total_pieces
        
        if self.is_complete and not was_complete:
            if self.agent_type in [AgentType.LEECHER, AgentType.HYBRID]:
                self.agent_type = AgentType.SEEDER
        
        elif not self.is_complete and len(self.file_pieces) > 0 and self.agent_type == AgentType.LEECHER:
            self.agent_type = AgentType.HYBRID
        
        # Clean up search tracking
        self.clear_found_pieces()
        
        return self.is_complete and not was_complete
    
    ########################################################
    # Helper Functions
    ########################################################

    # Get Needed Pieces
    def get_needed_pieces(self, total_pieces: int) -> Set[int]:
        """Get pieces this agent needs to complete the file."""
        return set(range(total_pieces)) - self.file_pieces
    
    # Get Retry Pieces
    def get_retry_pieces(self, current_round: int, ttl: int) -> Set[int]:
        """Get pieces that should be retried based on timeout of Original Query (2*ttl rounds)."""
        retry_pieces = set()
        timeout_rounds = 2 * ttl + 1
        
        for piece, search_round in list(self.last_search_time.items()):
            if current_round - search_round >= timeout_rounds:
                if piece not in self.file_pieces:
                    # Only mark as failed once when it first times out
                    if piece not in self.failed_pieces:
                        self.failed_pieces[piece] = 1
                        self.failed_piece_time[piece] = current_round  # Track when it failed
                        # Don't immediately retry - let it be excluded from searches
                        # retry_pieces.add(piece)  # REMOVED: Don't retry immediately
                # Remove from tracking regardless of whether it was found
                self.last_search_time.pop(piece, None)
        
        return retry_pieces
    
    # Decide Gossip Actions this node should take
    def decide_gossip_actions(self, graph: nx.Graph, total_pieces: int, rng: random.Random, K: int = 3, ttl: int = 5, current_round: int = 0, neighbor_selection: str = "bandwidth", single_agent: Optional[int] = None) -> None:
        """Decide whether to initiate gossip searches or respond to queries.
        Now using an Outbox Messaging setup rather than Simulation co-ordinated Messaging
        """
        # If we're a leecher or hybrid and need pieces, decide whether to search
        # Barebones, 30% chance to initiate a search if we need pieces
        # initiate a query up to K neighbors with TTL
        # add retry for pieces that have not been found after 2*ttl + 1 rounds
        if self.agent_type in [AgentType.LEECHER, AgentType.HYBRID]:
            can_initiate_search = (single_agent is None or single_agent == self.node_id)
            
            if can_initiate_search:
                needed_pieces = self.get_needed_pieces(total_pieces)
                retry_pieces = self.get_retry_pieces(current_round, ttl)  # This now only handles timeout cleanup
                failed_retry_pieces = self.get_failed_pieces_for_retry(current_round)  # Get pieces to retry
                
                # Available pieces = needed pieces - failed pieces - currently searching + retry pieces
                currently_searching = set(self.last_search_time.keys())
                available_pieces = (needed_pieces - set(self.failed_pieces.keys()) - currently_searching) | failed_retry_pieces
                
                if available_pieces:
                    search_prob = 0.3
                    if rng.random() < search_prob:
                        piece = rng.choice(list(available_pieces))                    
                        
                        # ie; could increase TTL for retry pieces - leave as static for now.
                        search_result = self.initiate_gossip_query(piece, ttl, K, graph, rng, neighbor_selection)
                        if search_result["forwards"]:  # Only track if search was successful
                            self.last_search_time[piece] = current_round
                            
                            # migrated from simulator.py
                            for target in search_result["forwards"]:
                                query_message = {
                                    "type": "query",
                                    "query_uuid": search_result["query_uuid"],
                                    "piece": search_result["piece"],
                                    "ttl": search_result["ttl"],
                                    "K": K,
                                    "from_node": self.node_id,
                                    "to_node": target,
                                    "origin": self.node_id
                                }
                                self.outbox.append(query_message)

    
    # Process Gossip Message
    def process_gossip_message(self, message: Dict, graph: nx.Graph, rng: random.Random, neighbor_selection: str = "bandwidth") -> None:
        """Process incoming gossip messages (queries or hits) and return response based on type."""
        if message["type"] == "query":
            # Process incoming if type is query
            result = self.process_gossip_query(
                message["query_uuid"], 
                message["piece"], 
                message["ttl"], 
                message["from_node"], 
                message["K"], 
                graph, 
                rng,
                neighbor_selection
            )
            
            for target in result["forwards"]:
                next_query = {
                    "type": "query",
                    "query_uuid": message["query_uuid"],
                    "piece": message["piece"],
                    "ttl": result["ttl"],  # Use the decremented TTL
                    "K": message["K"],
                    "from_node": self.node_id,
                    "to_node": target,
                    "origin": message["origin"]
                }
                if next_query["ttl"] > 0:  # Only forward if TTL > 0
                    self.outbox.append(next_query)
            
            # "If we have the piece", create a transfer (but check for duplicates)
            if result["hit"] is not None:
                hit_message = {
                    "type": "hit",
                    "query_uuid": message["query_uuid"],
                    "piece": message["piece"],
                    "from_node": self.node_id,
                    "to_node": message["from_node"],
                    "hit_node": result["hit"],
                    "origin": message["origin"]
                }
                self.outbox.append(hit_message)
        
        elif message["type"] == "hit":
            # Check if hit on origin?
            if message["to_node"] == message["origin"]:
                # Hit reached origin - create transfer
                piece = message["piece"]
                hit_node = message["hit_node"]
                
                # print(f"Agent {self.node_id} received hit for piece {piece} from {hit_node}")
                
                # Check if we don't already have this piece
                if piece not in self.file_pieces:
                    # print(f"Agent {self.node_id} creating transfer action for piece {piece}")
                    # Create transfer action (this will be processed by the simulator)
                    transfer_action = {
                        "type": "transfer",
                        "from": hit_node,
                        "to": self.node_id,
                        "to_node": self.node_id,
                        "piece": piece,
                        "query_uuid": message["query_uuid"]
                    }
                    self.outbox.append(transfer_action)
                    # print(f"Agent {self.node_id} outbox now has {len(self.outbox)} messages")
                    # print(f"Transfer action: {transfer_action}")
                else:
                    # print(f"Agent {self.node_id} already has piece {piece}")
                    pass
            else:
                # bck to Ori
                hit_result = self.process_gossip_hit(
                    message["query_uuid"], 
                    message["hit_node"], 
                    graph
                )
                
                # Forward Message
                for target in hit_result["forwards"]:
                    next_hit = {
                        "type": "hit",
                        "query_uuid": message["query_uuid"],
                        "piece": message["piece"],
                        "from_node": self.node_id,
                        "to_node": target,
                        "hit_node": message.get("hit_node", hit_result.get("hit_node")),
                        "origin": message["origin"]  # Always use the OG origin
                    }
                    self.outbox.append(next_hit)
    
    def process_gossip_query(self, query_uuid: str, piece: int, ttl: int, from_node: int, K: int, graph: nx.Graph, rng: random.Random, neighbor_selection: str = "bandwidth") -> Dict:
        """Process an incoming gossip query."""
        # Have we seen this query before? If so, ignore it and mark as duplicate
        if query_uuid in self.seen_queries:
            return {"query_uuid": query_uuid, "forwards": [], "hit": None, "duplicate": True, "ttl": ttl}
        
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
            
            if candidates:
                forwards = self.select_neighbors(candidates, K, graph, rng, neighbor_selection)
        
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
    
    # Initiate Gossip Query for a specific piece
    def initiate_gossip_query(self, piece: int, ttl: int, K: int, graph: nx.Graph, rng: random.Random, neighbor_selection: str = "bandwidth") -> Dict:
        """Initiate a gossip query for a specific piece."""
        query_uuid = str(uuid.uuid4()) # random
        self.seen_queries.add(query_uuid) # Mark as seen to avoid re-proc our own query
        
        # Choose K neighbors to forward to based on selection method
        neighbors = list(graph.neighbors(self.node_id))
        if not neighbors:
            return {"query_uuid": query_uuid, "forwards": [], "hit": None}
        
        selected_neighbors = self.select_neighbors(neighbors, K, graph, rng, neighbor_selection)
        
        return {
            "query_uuid": query_uuid,
            "piece": piece,
            "ttl": ttl,
            "forwards": selected_neighbors,
            "hit": None
        }
    
    # Get Link Quality (Edge Weight)
    def get_link_quality(self, graph: nx.Graph, neighbor: int) -> float:
        """Get the bandwidth between this node and neighbor."""
        if graph.has_edge(self.node_id, neighbor):
            return graph[self.node_id][neighbor].get('weight', 50.0)  # Default to 50 if no weight
        return 0.0  # No direct connection
    
    # Select Neighbor Method (random or bandwidth)
    def select_neighbors(self, neighbors: List[int], K: int, graph: nx.Graph, rng: random.Random, method: str = "bandwidth") -> List[int]:
        """Select K neighbors based on the specified method."""
        if len(neighbors) <= K:
            return neighbors
        
        if method == "random":
            # Pure random selection
            return rng.sample(neighbors, K)
        
        elif method == "bandwidth":
            # Bandwidth weight filtered, then top K
            weighted_neighbors = []
            for neighbor in neighbors:
                link_quality = self.get_link_quality(graph, neighbor)
                weighted_neighbors.append((neighbor, link_quality))
            
            # Sort by link quality - prefer better connections
            weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
            return [n for n, _ in weighted_neighbors[:K]]
        
        else:
            return rng.sample(neighbors, K)
    
    # Clear Old Queries
    def clear_old_queries(self, max_age: int = 100) -> None:
        """Clear old queries that are too old to exist"""
        if len(self.seen_queries) > max_age:
            self.seen_queries.clear()
            # Don't clear query_routing here as it's needed for hit routing
            # The clear_completed_queries function handles routing cleanup
    
    # Clear Completed Queries
    def clear_completed_queries(self, active_query_uuids: set) -> None:
        """Clear routing for queries that are no longer active."""
        completed_queries = set(self.query_routing.keys()) - active_query_uuids
        for query_uuid in completed_queries:
            self.query_routing.pop(query_uuid, None)
    
    # Clear Found Pieces from Search Tracking
    def clear_found_pieces(self) -> None:
        """Clear search tracking for pieces that have been found."""
        pieces_to_remove = []
        for piece in self.last_search_time.keys():
            if piece in self.file_pieces:
                pieces_to_remove.append(piece)
        
        for piece in pieces_to_remove:
            self.last_search_time.pop(piece, None)
            self.failed_pieces.pop(piece, None)  # Also clear retry count
    
    # Mark Failed Piece and Increment Retry Count for that piece
    def mark_failed_piece(self, piece: int) -> None:
        """Mark a piece as failed to be found (for retry)."""
        if piece in self.last_search_time:
            self.failed_pieces[piece] = self.failed_pieces.get(piece, 0) + 1
    
    # Get pieces that should be retried after waiting 5 rounds
    def get_failed_pieces_for_retry(self, current_round: int, retry_delay: int = 5) -> Set[int]:
        """Get failed pieces that should be retried after waiting 5 rounds."""
        retry_pieces = set()
        for piece, fail_count in self.failed_pieces.items():
            # Only retry up to 5 times to prevent infinite retries
            # maybe could do a exponential backoff
            if fail_count <= 5:
                failure_round = self.failed_piece_time.get(piece, 0)
                rounds_since_failure = current_round - failure_round

                if rounds_since_failure >= 5:
                    retry_pieces.add(piece)
        return retry_pieces
    
    # Kill node - store edges and set to DEAD
    def kill_node(self, graph: nx.Graph) -> None:
        """Kill this node - store edges and set to DEAD state."""
        if self.agent_type == AgentType.DEAD:
            return  # Already dead - RIP
            
        # Store original role
        self.original_role = self.agent_type
        
        # Store all edges
        self.stored_edges.clear()
        for neighbor in list(graph.neighbors(self.node_id)):
            edge_data = graph[self.node_id][neighbor].copy()
            self.stored_edges.append((neighbor, edge_data.get('weight', 50.0), edge_data))
        
        # Remove all edges from graph
        edges_to_remove = list(graph.edges(self.node_id))
        for edge in edges_to_remove:
            graph.remove_edge(*edge)
        
        # Set agent to DEAD
        self.agent_type = AgentType.DEAD
        
        # Clear all active state immediately to prevent any message sending
        self.inbox.clear()
        self.outbox.clear()
        self.seen_queries.clear()
        self.query_routing.clear()
    
    # Revive node
    def revive_node(self, graph: nx.Graph) -> None:
        """Revive this node."""
        if self.agent_type != AgentType.DEAD:
            return  # Not dead - yay!
            
        # Restore
        for neighbor, weight, attributes in self.stored_edges:
            # Remove weight from attributes if it exists to avoid duplicate
            edge_attrs = attributes.copy()
            edge_attrs.pop('weight', None)
            graph.add_edge(self.node_id, neighbor, weight=weight, **edge_attrs)

        if self.original_role:
            self.agent_type = self.original_role
            self.original_role = None
        else:
            self.agent_type = AgentType.LEECHER  # Default fallback should work
        
        # Clear stored edges
        self.stored_edges.clear()


########################################################
# Add Agent to Graph
########################################################
def add_node(G: nx.Graph, agent_type: AgentType, node_id: Optional[int] = None, connect_to_existing: bool = True, connection_prob: float = 0.3) -> Tuple[int, Agent]:
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

########################################################
# Remove Agent from Graph
########################################################
def remove_node(G: nx.Graph, node_id: int) -> bool:
    """
    Remove a node from the graph.
    """
    if node_id not in G.nodes():
        return False
    
    G.remove_node(node_id)
    return True

########################################################
# Assign N Seeders to Graph
########################################################
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

########################################################
# Initialize File Sharing and Piece Distribution
########################################################
def initialize_file_sharing(G: nx.Graph, file_size_pieces: int, seed: Optional[int] = None, distribution_type: str = "n_seeders", n_seeders: int = 1) -> None:
    """
    File sharing, give seeders all file pieces and leechers none as a test
    Could expand into other distributions later (custom, random, etc.)
    """
    rng = random.Random(seed)

    def set_agent_pieces_and_role(G, node_id, pieces, file_size_pieces):
        pieces_set = set(pieces)
        is_complete = len(pieces_set) >= file_size_pieces
        G.nodes[node_id]["file_pieces"] = pieces_set
        G.nodes[node_id]["is_complete"] = is_complete

        agent_obj = G.nodes[node_id].get("agent_object")
        if agent_obj:
            agent_obj.file_pieces = pieces_set
            agent_obj.is_complete = is_complete
            if is_complete:
                agent_obj.agent_type = AgentType.SEEDER
            elif len(pieces_set) > 0:
                agent_obj.agent_type = AgentType.HYBRID
            else:
                agent_obj.agent_type = AgentType.LEECHER
        if is_complete:
            G.nodes[node_id]["role"] = "seeder"
        elif len(pieces_set) > 0:
            G.nodes[node_id]["role"] = "hybrid"
        else:
            G.nodes[node_id]["role"] = "leecher"

    if distribution_type == "n_seeders":
        # For n_seeders, work with existing roles from assign_n_seeders
        for node, data in G.nodes(data=True):
            if "role" in data:
                agent_type = AgentType(data["role"])
                if agent_type == AgentType.SEEDER:
                    # Seeders start with all file pieces
                    set_agent_pieces_and_role(G, node, range(file_size_pieces), file_size_pieces)
                else:  # LEECHER
                    # Leechers start with no pieces
                    set_agent_pieces_and_role(G, node, set(), file_size_pieces)
    else:
        raise ValueError(f"Unknown distribution_type: {distribution_type}")

########################################################
# Get Agent Info
########################################################
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

########################################################
# Update Agent Completion & Role
# - If agent becomes complete, convert to seeder
# - If agent has some pieces but not all, convert to hybrid
########################################################
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

########################################################
# Decide Gossip Actions at destination Node (Simulation Layer)
########################################################
def process_gossip_message_at_node(G: nx.Graph, node_id: int, message: Dict, 
                                  seed: Optional[int] = None, neighbor_selection: str = "bandwidth") -> Dict:
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
    
    return agent_obj.process_gossip_message(message, G, rng, neighbor_selection)


########################################################
# Reset Agent State
########################################################
def reset_agent_state(agent: Agent) -> None:
    """Reset an agent to its initial state."""
    agent.file_pieces.clear()
    agent.is_complete = False
    agent.seen_queries.clear()
    agent.query_routing.clear()
    agent.failed_pieces.clear()
    agent.last_search_time.clear()
    agent.failed_piece_time.clear()
    agent.inbox.clear()
    agent.outbox.clear()
    agent.stored_edges.clear()
    agent.original_role = None
    agent.agent_type = AgentType.LEECHER

########################################################
# Get Network Stats
########################################################
def get_network_stats(G: nx.Graph, total_pieces: int) -> Dict:
    """
    Overview of the network
    """
    seeders = [node for node in G.nodes() if G.nodes[node].get("role") == "seeder"]
    leechers = [node for node in G.nodes() if G.nodes[node].get("role") == "leecher"]
    hybrids = [node for node in G.nodes() if G.nodes[node].get("role") == "hybrid"]
    dead_nodes = [node for node in G.nodes() if G.nodes[node].get("role") == "dead"]
    complete_leechers = [node for node in leechers if G.nodes[node].get("is_complete", False)]
    complete_hybrids = [node for node in hybrids if G.nodes[node].get("is_complete", False)]
    
    total_pieces_in_network = sum(len(G.nodes[node].get("file_pieces", set())) for node in G.nodes())
    
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "seeders": len(seeders),
        "leechers": len(leechers),
        "hybrids": len(hybrids),
        "dead_nodes": len(dead_nodes),
        "complete_leechers": len(complete_leechers),
        "complete_hybrids": len(complete_hybrids),
        "incomplete_leechers": len(leechers) - len(complete_leechers),
        "incomplete_hybrids": len(hybrids) - len(complete_hybrids),
        "total_pieces_in_network": total_pieces_in_network,
        "completion_rate": (len(seeders) + len(complete_leechers) + len(complete_hybrids)) / (G.number_of_nodes() - len(dead_nodes)) if (G.number_of_nodes() - len(dead_nodes)) > 0 else 0
    }