import matplotlib.pyplot as plt
import networkx as nx
from src.graph import nxgraph
from src.analytics import clustering_co, path_length
from utils.plotter import plot_activity_over_time
from widgets.init_widget import set_graph_data
import src.agent as agent_module
from src.agent import AgentType
from utils.simulator_new import simulate_round_agent_driven, get_network_stats
from typing import Dict, List


def identify_hubs(graph: nx.Graph, top_k: int = 2) -> List[int]:
    """top-k highest degree nodes (hubs) in the graph."""
    degree_dict = dict(graph.degree())
    sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    return [node_id for node_id, _ in sorted_nodes[:top_k]]


def create_hub_removal_scenario(graph: nx.Graph, kill_round: int = 10) -> Dict:
    hubs = identify_hubs(graph, top_k=2)
    
    return {
        'type': 'kill_peers',
        'rips': [
            {'id': hubs[0], 'kill_r': kill_round, 'rev_r': None},
            {'id': hubs[1], 'kill_r': kill_round, 'rev_r': None}
        ]
    }


def create_lifecycle_scenario(graph: nx.Graph) -> Dict:
    hubs = identify_hubs(graph, top_k=2)
    
    return {
        'type': 'kill_peers',
        'rips': [
            {'id': hubs[0], 'kill_r': 10, 'rev_r': 30},  # 20 round gap
            {'id': hubs[1], 'kill_r': 20, 'rev_r': 50}   # 30 round gap
        ]
    }


def run_resilience_test():
    G = nxgraph()
    
    seed = 42
    nodes = 20
    lower_ut = 5
    upper_ut = 100
    FILE_PIECES = 15
    n_seeders = 3
    search_mode = 'realistic'
    neighbor_selection = 'random'
    ttl = 5
    weighted = True
    
    test_graphs = []
    
    for m in range(1, 6):
        ba = G.BA_graph(nodes=nodes, edges=m, seed=seed, weighted=weighted, 
                       lower_ut=lower_ut, upper_ut=upper_ut)
        test_graphs.append(('BA', m, ba))

    for m in range(1, 6):
        avg_deg = 2 * m  # BA graph with m edges has avg degree ~2m
        m_er = int(nodes * avg_deg / 2)
        er = G.ER_Graph_nm(nodes=nodes, edges=m_er, weighted=weighted, seed=seed,
                          lower_ut=lower_ut, upper_ut=upper_ut)
        test_graphs.append(('ER', m_er, er))
        
        # # Create ER graph with exact same edge count as BA
        # E_ba = ba.number_of_edges()
        # er = G.ER_Graph_nm(nodes=nodes, edges=E_ba, weighted=weighted, seed=seed,
        #                   lower_ut=lower_ut, upper_ut=upper_ut)
        # test_graphs.append(('ER', E_ba, er))
    
    scenarios = [
        ('Hub Removal', create_hub_removal_scenario),
        ('Lifecycle', create_lifecycle_scenario)
    ]
    
    for scenario_name, scenario_func in scenarios:
        print(f"TESTING SCENARIO: {scenario_name}")
        
        for graph_type, param, graph in test_graphs:
            print(f"\nTesting {graph_type} graph (param={param})")
            
            # Copy of graph for re-test
            import copy
            graph_copy = copy.deepcopy(graph)
            
            set_graph_data(graph_copy, FILE_PIECES)
            agent_module.assign_n_seeders(graph_copy, n=n_seeders, seed=seed)
            agent_module.initialize_file_sharing(graph_copy, FILE_PIECES, seed=seed,
                                        distribution_type='n_seeders', n_seeders=n_seeders)
            
            scenario_data = scenario_func(graph_copy)
            print(f"Scenario: {scenario_data}")
            
            simulation_data = []
            max_rounds = 150
            
            for round_num in range(max_rounds):
                result = simulate_round_agent_driven(
                    graph_copy, FILE_PIECES, seed=seed + round_num,
                    cleanup_completed_queries=True, search_mode=search_mode,
                    current_round=round_num, neighbor_selection=neighbor_selection,
                    single_agent=None, scenario_data=scenario_data
                )
                
                simulation_data.append(result)
                
                stats = get_network_stats(graph_copy, FILE_PIECES)
                if stats['completion_rate'] >= 1.0:
                    print(f"  Completed in {round_num} rounds")
                    break
                elif round_num == max_rounds - 1:  # Last round
                    print(f"  Failed to complete - final completion rate: {stats['completion_rate']:.1%}")
            
            # # Generate and save
            import os
            from datetime import datetime
            
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_dir = f"data/resilience_test_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate plot without showing it
            plot_activity_over_time(
                simulation_data, 
                graph_copy, 
                f"{scenario_name} - {graph_type} Graph (param={param})",
                show_plot=False
            )
            
            #Save the plot
            filename = f"{scenario_name.lower().replace(' ', '_')}_{graph_type.lower()}_param_{param}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Saved plot: {filepath}")
            plt.close()


if __name__ == "__main__":
    run_resilience_test()
