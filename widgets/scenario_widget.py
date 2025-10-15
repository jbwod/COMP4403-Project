import ipywidgets as widgets
from IPython.display import display, clear_output

# Global var for scenario data
scenario_data = {
    'type': 'normal',
    'rips': []
}

def create_widgets():
    """Create scenario configuration widgets."""
    # Scenario type selector
    scenario_type = widgets.Dropdown(
        options=[
            ('Normal', 'normal'),
            ('Kill Peers', 'kill_peers'),
        ],
        value='normal',
        description='Scenario:',
        style={'description_width': 'initial'}
    )
    
    #  text input
    rip_list = widgets.Textarea(
        value='',
        placeholder='Enter kill/revive as an array:\n[{"id": 1, "kill_r": 5, "rev_r": 10}, {"id": 2, "kill_r": 8, "rev_r": null}]',
        description='Node RIP List:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px', height='100px')
    )
    
    # Set scenario button
    set_btn = widgets.Button(
        description='Set Scenario',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    # Degree
    degree_btn = widgets.Button(
        description='Analyze Node Degrees',
        button_style='info',
        layout=widgets.Layout(width='200px')
    )
    
    # Output area
    output_area = widgets.Output()
    
    return (scenario_type, rip_list, set_btn, degree_btn, output_area)

def on_set_clicked(b, scenario_type, rip_list, output_area):
    """Handle set scenario button click."""
    global scenario_data
    
    b.description = "Setting..."
    b.disabled = True
    
    try:
        with output_area:
            clear_output(wait=True)
            print("Setting scenario...")
            print(f"Selected scenario: {scenario_type.value}")
            
            if scenario_type.value == 'kill_peers':
                import json
                try:
                    rips = json.loads(rip_list.value) if rip_list.value.strip() else []
                    print(f"Node RIP list: {rips}")
                    scenario_data = {
                        'type': 'kill_peers',
                        'rips': rips
                    }
                    print("Kill peers scenario set successfully!")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    print("Please enter valid JSON format")
            else:
                scenario_data = {
                    'type': 'normal',
                    'rips': []
                }
                print("Normal scenario set successfully!")
            
    except Exception as e:
        print(f"Error setting scenario: {e}")
    finally:
        b.description = "Set Scenario"
        b.disabled = False

def on_degree_clicked(b, output_area):
    b.disabled = True
    
    try:
        with output_area:
            clear_output(wait=True)
            from widgets.init_widget import get_graph_data
            G, FILE_PIECES = get_graph_data()
            
            if G is None:
                print("Error: No graph data available. Generate a graph first.")
                return
            
            # Calculate degree
            degrees = dict(G.degree())
            
            if not degrees:
                print("No nodes found in the graph.")
                return
            
            # Find nodes with highest degree
            max_degree = max(degrees.values())
            highest_degree_nodes = [node for node, degree in degrees.items() if degree == max_degree]
            
            # Sort all nodes
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            
            print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            print(f"Maximum degree: {max_degree}")
            print(f"Nodes with highest degree ({max_degree}): {highest_degree_nodes}")
            print(f"\nAll nodes sorted by degree (descending):")
            
            for i, (node, degree) in enumerate(sorted_nodes):
                print(f"  {i+1:2d}. Node {node:2d}: degree {degree}")
            
            # misc
            avg_degree = sum(degrees.values()) / len(degrees)
            min_degree = min(degrees.values())
            print(f"\nDegree statistics:")
            print(f"  Average degree: {avg_degree:.2f}")
            print(f"  Minimum degree: {min_degree}")
            print(f"  Maximum degree: {max_degree}")
            print(f"  Degree range: {max_degree - min_degree}")
            
    except Exception as e:
        print(f"Error analyzing degrees: {e}")
        import traceback
        traceback.print_exc()
    finally:
        b.description = "Analyze Node Degrees"
        b.disabled = False

def display_scenario_widgets():
    """Display the scenario widgets."""

    (scenario_type, rip_list, set_btn, degree_btn, output_area) = create_widgets()
    
    def on_set_click(b):
        on_set_clicked(b, scenario_type, rip_list, output_area)
    
    def on_degree_click(b):
        on_degree_clicked(b, output_area)
    
    set_btn.on_click(on_set_click)
    degree_btn.on_click(on_degree_click)
    
    kill_peers_config = widgets.VBox([
        widgets.HTML("<h3>Kill Peers Configuration</h3>"),
        widgets.HTML("""
        <p> Nodes with higher degree are considered Hubs </p>
        <code>[{"id": node_id, "kill_r": kill_round, "rev_r": revive_round|null}]</code><br><br>
        """),
        rip_list,
        widgets.HTML("<br>"),
        degree_btn
    ])
    
    def update_config_visibility(change):
        if change['new'] == 'kill_peers':
            kill_peers_config.layout.display = 'block'
        else:
            kill_peers_config.layout.display = 'none'
    
    kill_peers_config.layout.display = 'none' if scenario_type.value != 'kill_peers' else 'block'
    
    scenario_type.observe(update_config_visibility, names='value')
    
    # Layouts
    controls = widgets.VBox([
        widgets.HTML("<h2>Scenario Configuration</h2>"),
        widgets.HBox([scenario_type, set_btn]),
        kill_peers_config,
        widgets.HTML("<hr>"),
        output_area
    ])
    
    display(controls)
    print("Configure your scenario and click 'Set Scenario' to apply.")

def get_scenario_data():
    """Get the current scenario data."""
    return scenario_data
