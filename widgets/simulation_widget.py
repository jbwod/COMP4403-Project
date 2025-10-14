import ipywidgets as widgets
from IPython.display import display, clear_output
import src.agent as agent_module
from utils.simulator_new import simulate_round_agent_driven, get_network_stats
from utils.plotter import draw_gossip_step_by_step, start_new_run, plot_activity_over_time

def create_widgets():
    """Create simulation configuration widgets."""
    # Simulation type selector
    simulation_type = widgets.Dropdown(
        options=[
            ('Run', 'run'),
        ],
        value='run',
        description='Simulation:',
        style={'description_width': 'initial'}
    )
    
    # Simulation parameters
    max_rounds = widgets.IntSlider(
        value=70,
        min=10,
        max=200,
        step=10,
        description='Max Rounds:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    seed = widgets.IntSlider(
        value=42,
        min=0,
        max=1000,
        step=1,
        description='Seed:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    search_mode = widgets.Dropdown(
        options=[
            ('Single', 'single'),
            ('Realistic', 'realistic')
        ],
        value='realistic',
        description='Search Mode:',
        style={'description_width': 'initial'}
    )
    
    neighbor_selection = widgets.Dropdown(
        options=[
            ('Bandwidth', 'bandwidth'),
            ('Random', 'random')
        ],
        value='random',
        description='Neighbor Selection:',
        style={'description_width': 'initial'}
    )
    
    ttl = widgets.IntSlider(
        value=5,
        min=1,
        max=10,
        step=1,
        description='TTL:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    k = widgets.IntSlider(
        value=3,
        min=1,
        max=10,
        step=1,
        description='K (neighbors):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    
    single_agent = widgets.IntText(
        value=0,
        min=0,
        max=100,
        description='Single Agent (0=all):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    save_images = widgets.Checkbox(
        value=False,
        description='Save Images',
        style={'description_width': 'initial'}
    )
    
    debug_output = widgets.Checkbox(
        value=False,
        description='Debug Output',
        style={'description_width': 'initial'}
    )
    
    visualize_output = widgets.Checkbox(
        value=True,
        description='Visualize Output',
        style={'description_width': 'initial'}
    )
    
    show_analytics = widgets.Checkbox(
        value=False,
        description='Show Analytics',
        style={'description_width': 'initial'}
    )
    
    # Run simulation button
    run_btn = widgets.Button(
        description='Run Simulation',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    # Output area
    output_area = widgets.Output()
    print('this is output area %s' % output_area)
    
    return (simulation_type, max_rounds, seed, search_mode, neighbor_selection, 
            ttl, k, single_agent, save_images, debug_output, visualize_output, show_analytics, run_btn, output_area)

def on_run_clicked(b, simulation_type, max_rounds, seed, search_mode, neighbor_selection, 
                  ttl, k, single_agent, save_images, debug_output, visualize_output, show_analytics, output_area):
    """Handle run simulation button click."""
    b.description = "Running..."
    b.disabled = True 
    try:
        with output_area:
            print('in output area')
            clear_output(wait=True)
            print("Running simulation...")
            
            # Get graph data from init widget
            from widgets.init_widget import get_graph_data
            G, FILE_PIECES = get_graph_data()
            
            if G is None or FILE_PIECES is None:
                print("Error: No graph data available. Generate a graph and initialize file sharing first.")
                return
            
            # G is the NetworkX graph from get_graph_data()
            print(f"Starting simulation with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Initialize simulation
            stats = get_network_stats(G, FILE_PIECES)
            print(f"Initial: {stats['incomplete_leechers']} leechers incomplete, {stats['completion_rate']:.1%} completion rate")
            
            if debug_output.value:
                print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                print(f"File pieces: {FILE_PIECES}")
                print(f"Max rounds: {max_rounds.value}")
                print(f"Search mode: {search_mode.value}")
                print(f"Neighbor selection: {neighbor_selection.value}")
                print(f"TTL: {ttl.value}")
                print(f"K (neighbors): {k.value}")
                print(f"Single agent: {single_agent.value if single_agent.value > 0 else 'All agents'}")
                print(f"Save images: {save_images.value}")
                print(f"Visualize output: {visualize_output.value}")

                print(f"Seeders: {stats['seeders']}")
                print(f"Leechers: {stats['leechers']} (incomplete: {stats['incomplete_leechers']})")
                print(f"Hybrids: {stats['hybrids']} (incomplete: {stats['incomplete_hybrids']})")
                print(f"Total pieces in network: {stats['total_pieces_in_network']}")
                print(f"Completion rate: {stats['completion_rate']:.1%}")
                for node in G.nodes():
                    agent = G.nodes[node].get('agent_object')
                    if agent:
                        pieces_count = len(agent.file_pieces)
                        needed = len(agent.get_needed_pieces(FILE_PIECES))
                        role = agent.agent_type.value
                        print(f"  Node {node}: {role} - {pieces_count}/{FILE_PIECES} pieces (needs {needed})")
            
            output_dir = start_new_run()
            
            retry_stats = {
                'total_retries': 0,
                'pieces_retried': set(),
                'rounds_to_completion': {},
                'previous_failed_pieces': {}
            }

            number_of_q = 0
            number_of_h = 0
            number_of_messages = 0
            number_of_transfers = 0
            
            # Collect simulation data for analytics
            simulation_data = []
            
            # Run simulation rounds
            for round_num in range(1, max_rounds.value + 1):
                if debug_output.value:
                    print(f"\n Round {round_num}")
                
                # Run one round of simulation
                result = simulate_round_agent_driven(
                    G, FILE_PIECES, 
                    seed=seed.value + round_num, 
                    cleanup_completed_queries=True, 
                    search_mode=search_mode.value, 
                    current_round=round_num, 
                    neighbor_selection=neighbor_selection.value,
                    single_agent=single_agent.value if single_agent.value > 0 else None
                )
                
                if debug_output.value:
                    print(f"\nROUND {round_num} RESULTS")
                    print(f"Total Messages: {result['total_messages']}")
                    print(f"Total Transfers: {result['total_transfers']}")
                    print(f"New Completions: {result['new_completions']}")
                    
                    # Debug message details
                    if result['message_rounds']:
                        queries, hits, _ = result['message_rounds']
                        if queries:
                            print(f"QUERIES: {len(queries)} messages")
                            for i, msg in enumerate(queries[:3]):  # Show first 3 queries
                                print(f"  Query {i+1}: {msg['from_node']} -> {msg['to_node']}, piece {msg['piece']}, TTL {msg['ttl']}")
                            if len(queries) > 3:
                                print(f"  ... and {len(queries) - 3} more queries")
                        
                        if hits:
                            print(f"HITS: {len(hits)} messages")
                            for i, msg in enumerate(hits[:3]):  # Show first 3 hits
                                print(f"  Hit {i+1}: {msg['from_node']} -> {msg['to_node']}, piece {msg['piece']}, hit_node {msg['hit_node']}")
                            if len(hits) > 3:
                                print(f"  ... and {len(hits) - 3} more hits")
                    
                    if result['transfers']:
                        for transfer in result['transfers']:
                            print(f"  Piece {transfer['piece']}: Node {transfer['from']} -> Node {transfer['to']}")
                
                if result['new_completions']:
                    for node in result['new_completions']:
                        retry_stats['rounds_to_completion'][node] = round_num
                        if debug_output.value:
                            print(f"\nNODE {node} COMPLETED THE FILE!")
                
                # Track when pieces fail (timeout)
                for node in G.nodes():
                    agent = G.nodes[node].get('agent_object')
                    if agent and agent.failed_pieces:
                        prev_failed = retry_stats['previous_failed_pieces'].get(node, {})
                        for piece, fail_count in agent.failed_pieces.items():
                            prev_count = prev_failed.get(piece, 0)
                            if fail_count > prev_count:
                                new_failures = fail_count - prev_count
                                retry_stats['total_retries'] += new_failures
                                retry_stats['pieces_retried'].add(piece)
                                if debug_output.value:
                                    print(f"  Node {node} failed piece {piece} {new_failures} times (total failures: {fail_count})")
                        
                        # Update previous state
                        retry_stats['previous_failed_pieces'][node] = dict(agent.failed_pieces)

                number_of_h += len(result['message_rounds'][1])
                number_of_q += len(result['message_rounds'][0])
                number_of_messages += result.get('total_messages', 0)
                number_of_transfers += result.get('total_transfers', 0)
                if show_analytics.value:
                    simulation_data.append(result)
                # Debug agent states
                if debug_output.value:
                    print(f"\n--- AGENT STATES ---")
                    for node in G.nodes():
                        agent = G.nodes[node].get('agent_object')
                        if agent:
                            pieces_count = len(agent.file_pieces)
                            needed = len(agent.get_needed_pieces(FILE_PIECES))
                            role = agent.agent_type.value
                            print(f"  Node {node}: {role} - {pieces_count}/{FILE_PIECES} pieces (needs {needed})")
                            if agent.failed_pieces:
                                print(f"    Failed pieces: {list(agent.failed_pieces.keys())}")
                            if agent.last_search_time:
                                print(f"    Currently searching: {list(agent.last_search_time.keys())}")
                
                stats = get_network_stats(G, FILE_PIECES)
                if debug_output.value:
                    print(f"Completion Rate: {stats['completion_rate']:.1%}")
                    print(f"Seeders: {stats['seeders']}")
                    print(f"Leechers: {stats['leechers']} (incomplete: {stats['incomplete_leechers']})")
                    print(f"Hybrids: {stats['hybrids']} (incomplete: {stats['incomplete_hybrids']})")
                    print(f"Total pieces in network: {stats['total_pieces_in_network']}")
                #print(result['message_rounds'])
                # Visualize if enabled
                if visualize_output.value:
                    show_debug_info = False # used to show the node brain
                    draw_gossip_step_by_step(G, result['message_rounds'], result['transfers'], 
                                            FILE_PIECES, round_num, save_images=save_images.value, max_ttl=ttl.value, show_debug_info=show_debug_info)
                
                # Check for completion
                if stats['completion_rate'] >= 1.0:
                    print(f"\nAll nodes have all pieces in {round_num} rounds")
                    break
            
            # Final statistics
            print(f"\n{'='*60}")
            print(f"SIMULATION COMPLETED!")
            print(f"{'='*60}")
            print(f"Total failures across all agents: {retry_stats['total_retries']}")
            print(f"Unique pieces that failed: {len(retry_stats['pieces_retried'])}")
            print(f"Pieces that failed: {sorted(retry_stats['pieces_retried'])}")
            if retry_stats['rounds_to_completion']:
                avg_rounds = sum(retry_stats['rounds_to_completion'].values()) / len(retry_stats['rounds_to_completion'])
                print(f"Average rounds to completion: {avg_rounds:.1f}")
                print(f"Completion times: {retry_stats['rounds_to_completion']}")
            
            final_stats = get_network_stats(G, FILE_PIECES)
            print(f"Total nodes: {final_stats['total_nodes']}")
            print(f"Total edges: {final_stats['total_edges']}")
            print(f"Seeders: {final_stats['seeders']}")
            print(f"Leechers: {final_stats['leechers']}")
            print(f"Complete leechers: {final_stats['complete_leechers']}")
            print(f"Incomplete leechers: {final_stats['incomplete_leechers']}")
            print(f"Completion rate: {final_stats['completion_rate']:.1%}")
            print(f"Total pieces in network: {final_stats['total_pieces_in_network']}")
            
            if debug_output.value:
                for node in G.nodes():
                    agent = G.nodes[node].get('agent_object')
                    if agent:
                        pieces_count = len(agent.file_pieces)
                        needed = len(agent.get_needed_pieces(FILE_PIECES))
                        role = agent.agent_type.value
                        print(f"  Node {node}: {role} - {pieces_count}/{FILE_PIECES} pieces (needs {needed})")
                        if agent.failed_pieces:
                            print(f"    Failed pieces: {list(agent.failed_pieces.keys())}")
                        if agent.last_search_time:
                            print(f"    Currently searching: {list(agent.last_search_time.keys())}")
                
                print(f"Rounds completed: {round_num}")
                print(f"Total queries sent: {number_of_q}")
                print(f"Total hits sent: {number_of_h}")
                print(f"Total messages sent: {number_of_messages}")
                print(f"Total transfers completed: {number_of_transfers}")
                print(f"Success rate: {final_stats['completion_rate']:.1%}")
            
            # Display analytics if enabled
            if show_analytics.value and simulation_data:
                print(f"\n{'='*60}")
                print("GENERATING ANALYTICS...")
                print(f"{'='*60}")
                
                try:
                    # Activity over time
                    plot_activity_over_time(simulation_data, "Simulation Activity Over Time")
                    
                except Exception as e:
                    print(f"Error generating analytics: {e}")
                    import traceback
                    traceback.print_exc()
            
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        b.description = "Run Simulation"
        b.disabled = False


def display_simulation_widgets():
    """Display the simulation widgets."""

    (simulation_type, max_rounds, seed, search_mode, neighbor_selection, 
     ttl, k, single_agent, save_images, debug_output, visualize_output, show_analytics, run_btn, output_area) = create_widgets()
    
    def on_run_click(b):
        on_run_clicked(b, simulation_type, max_rounds, seed, search_mode, neighbor_selection, 
                      ttl, k, single_agent, save_images, debug_output, visualize_output, show_analytics, output_area)
    
    run_btn.on_click(on_run_click)
    
    # Layouts
    controls = widgets.VBox([
        widgets.HTML("<h2>Simulation Control</h2>"),
        widgets.HBox([simulation_type, run_btn]),
        widgets.HTML("<hr>"),
        widgets.HTML("<h3>Simulation Parameters</h3>"),
        widgets.HBox([max_rounds, seed]),
        widgets.HBox([search_mode, neighbor_selection]),
        widgets.HBox([ttl, k]),
        widgets.HBox([single_agent]),
        widgets.HBox([save_images, debug_output, visualize_output, show_analytics]),
        widgets.HTML("<hr>"),
        output_area
    ])
    
    display(controls)
    print("Generate a graph and initialize file sharing first.")
